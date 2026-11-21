#include "FemMesh.h"

#include "picosha2.h"

#include <CDT.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace omfdm {
namespace fem {

namespace {

// Convert a 3-channel RGB value to a key.
static uint32_t rgbKey(const cv::Vec3b& p) {
    return (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 8) | uint32_t(p[2]);
}

// Point-in-polygon (ray casting), metres.
static bool pointInPolygon(const std::vector<Point2D>& poly, double x, double y) {
    bool inside = false;
    size_t j = poly.size() ? poly.size() - 1 : 0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Point2D& a = poly[i];
        const Point2D& b = poly[j];
        bool cond = ((a.y > y) != (b.y > y)) &&
                    (x < (b.x - a.x) * (y - a.y) / (b.y - a.y + 1e-30) + a.x);
        if (cond) inside = !inside;
        j = i;
    }
    return inside;
}

// Douglas-Peucker.
static double perpDist(const Point2D& p, const Point2D& a, const Point2D& b) {
    double dx = b.x - a.x, dy = b.y - a.y;
    double len = std::hypot(dx, dy);
    if (len < 1e-30) return std::hypot(p.x - a.x, p.y - a.y);
    return std::fabs(dy * (p.x - a.x) - dx * (p.y - a.y)) / len;
}

static void dpRecurse(const std::vector<Point2D>& pts, size_t first, size_t last,
                      double tol, std::vector<char>& keep) {
    double maxD = 0; size_t idx = 0;
    for (size_t i = first + 1; i < last; ++i) {
        double d = perpDist(pts[i], pts[first], pts[last]);
        if (d > maxD) { maxD = d; idx = i; }
    }
    if (maxD > tol && idx != 0) {
        keep[idx] = 1;
        dpRecurse(pts, first, idx, tol, keep);
        dpRecurse(pts, idx, last, tol, keep);
    }
}

static std::vector<Point2D> douglasPeucker(const std::vector<Point2D>& in, double tol) {
    if (in.size() < 3) return in;
    std::vector<char> keep(in.size(), 0);
    keep.front() = 1; keep.back() = 1;
    dpRecurse(in, 0, in.size() - 1, tol, keep);
    std::vector<Point2D> out;
    for (size_t i = 0; i < in.size(); ++i) if (keep[i]) out.push_back(in[i]);
    if (out.size() < 3) return in;
    return out;
}

static double arcLen(const std::vector<Point2D>& p) {
    double s = 0;
    for (size_t i = 1; i < p.size(); ++i) s += std::hypot(p[i].x - p[i-1].x, p[i].y - p[i-1].y);
    return s;
}

// Resample a closed polyline to uniform spacing (arc-length parametrised).
static std::vector<Point2D> resampleClosed(const std::vector<Point2D>& in, double spacing) {
    if (in.size() < 3 || spacing <= 0) return in;
    const size_t n = in.size();
    std::vector<double> cum(n + 1, 0.0); // cum[k] = arc length from in[0] to in[k]; cum[n] = perimeter
    for (size_t k = 1; k <= n; ++k) {
        const Point2D& a = in[k - 1];
        const Point2D& b = in[k % n];
        cum[k] = cum[k - 1] + std::hypot(b.x - a.x, b.y - a.y);
    }
    const double total = cum[n];
    if (total < 1e-30) return in;
    const int count = std::max(3, (int)std::round(total / spacing));
    std::vector<Point2D> out;
    out.reserve(count);
    size_t seg = 0;
    for (int k = 0; k < count; ++k) {
        const double s = (double)k / count * total;
        while (seg + 1 < n && cum[seg + 1] < s) ++seg;
        const double segLen = cum[seg + 1] - cum[seg];
        const double t = segLen > 1e-30 ? (s - cum[seg]) / segLen : 0.0;
        const Point2D& a = in[seg];
        const Point2D& b = in[(seg + 1) % n];
        out.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)});
    }
    return out;
}


static double minAngleDeg(const std::vector<Point2D>& tri) {
    // triangle vertices tri[0..2]
    auto ang = [&](int i, int j, int k) {
        double ax = tri[j].x - tri[i].x, ay = tri[j].y - tri[i].y;
        double bx = tri[k].x - tri[i].x, by = tri[k].y - tri[i].y;
        double dot = ax*bx + ay*by;
        double la = std::hypot(ax, ay), lb = std::hypot(bx, by);
        if (la < 1e-30 || lb < 1e-30) return 0.0;
        double c = dot / (la * lb);
        c = std::max(-1.0, std::min(1.0, c));
        return std::acos(c) * 180.0 / M_PI;
    };
    double a1 = ang(0,1,2);
    double a2 = ang(1,0,2);
    double a3 = ang(2,0,1);
    return std::min({a1, a2, a3});
}

} // anon

// ---------- buildMask ----------
std::vector<int> MeshBuilder::buildMask(const cv::Mat& imageRgb,
                                        const std::vector<MaterialDesc>& materials) {
    CV_Assert(imageRgb.type() == CV_8UC3);
    int nx = imageRgb.cols;
    int ny = imageRgb.rows;
    std::vector<int> mask((size_t)nx * ny, 0);
    // map rgb -> material index
    std::map<uint32_t, int> rgbToMat;
    for (size_t m = 0; m < materials.size(); ++m) {
        const auto& md = materials[m];
        uint32_t k = (uint32_t(md.rgb[0]) << 16) | (uint32_t(md.rgb[1]) << 8) | md.rgb[2];
        rgbToMat[k] = (int)m + 1; // 0 reserved for air/unmatched; material index 1-based
    }
    // Actually use 0-based material indices internally; air = -1 sentinel -> 0.
    // Redo: mask stores material index (0 = first material). Unmatched = -1 -> store as materials.size() (air).
    // Simpler: mask value = material index or -1 for air. Store -1 as -1.
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            cv::Vec3b px = imageRgb.at<cv::Vec3b>(y, x);
            uint32_t k = (uint32_t(px[0]) << 16) | (uint32_t(px[1]) << 8) | px[2];
            auto it = rgbToMat.find(k);
            mask[(size_t)y * nx + x] = (it != rgbToMat.end()) ? it->second - 1 : -1;
        }
    }
    // material 0 is air if not declared; ensure unmatched -> a special air index.
    // We'll remap: -1 means air. Keep -1.
    return mask;
}

// ---------- extractContours ----------
std::vector<Contour> MeshBuilder::extractContours(const std::vector<int>& mask,
                                                  int nx, int ny,
                                                  double dx, double dy) {
    std::vector<Contour> result;
    if (nx <= 0 || ny <= 0 || (int)mask.size() != nx * ny) return result;

    // Build a set of material ids present.
    std::set<int> mats;
    for (int v : mask) mats.insert(v);
    // For each material id >= 0, extract its contours via OpenCV on a binary mask.
    // Holes come free from RETR_TREE.
    for (int mid : mats) {
        if (mid < 0) continue; // air handled as background
        cv::Mat bin(ny, nx, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                if (mask[(size_t)y * nx + x] == mid) bin.at<uint8_t>(y, x) = 255;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bin, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        for (size_t ci = 0; ci < contours.size(); ++ci) {
            if (contours[ci].size() < 3) continue;
            Contour c;
            for (const auto& pt : contours[ci]) {
                // mask is already flipped (Y-up); image coords -> metres.
                c.points.push_back({pt.x * dx, pt.y * dy});
            }
            c.materialInside = mid;
            // material outside = sample a point just outside the first vertex normal; approximate by air (-1-> materials.size())
            c.materialOutside = -1;
            // hole detection: if hierarchy parent != -1 it's a hole
            if (ci < hierarchy.size() && hierarchy[ci][3] != -1) c.isHole = true;
            result.push_back(std::move(c));
        }
    }
    return result;
}

// ---------- simplifyContour ----------
std::vector<Point2D> MeshBuilder::simplifyContour(const std::vector<Point2D>& in,
                                                  double tolerance,
                                                  double minSpacing,
                                                  double maxSpacing) {
    if (in.size() < 3) return in;
    // Remove consecutive duplicates.
    std::vector<Point2D> dedup;
    for (const auto& p : in) {
        if (dedup.empty() || std::hypot(p.x-dedup.back().x, p.y-dedup.back().y) > 1e-12) dedup.push_back(p);
    }
    if (dedup.size() < 3) return in;
    // Remove collinear middle points.
    std::vector<Point2D> clean;
    clean.push_back(dedup[0]);
    for (size_t i = 1; i + 1 < dedup.size(); ++i) {
        const Point2D& a = clean.back();
        const Point2D& b = dedup[i];
        const Point2D& c = dedup[i+1];
        double cross = (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
        if (std::fabs(cross) > 1e-12) clean.push_back(b);
    }
    clean.push_back(dedup.back());
    if (clean.size() < 3) clean = dedup;

    double tol = tolerance > 0 ? tolerance : 1e-12;
    auto dp = douglasPeucker(clean, tol);
    double L = arcLen(dp);
    if (L < 1e-30) return dp;
    double spacing = minSpacing > 0 ? minSpacing : (maxSpacing > 0 ? maxSpacing * 0.5 : L / (double)dp.size());
    if (maxSpacing > 0 && spacing > maxSpacing) spacing = maxSpacing;
    if (minSpacing > 0 && spacing < minSpacing) spacing = minSpacing;
    auto rs = resampleClosed(dp, spacing);
    if (rs.size() < 3) return dp;
    return rs;
}

// ---------- build ----------
FemMesh MeshBuilder::build(const cv::Mat& imageRgb,
                           const std::vector<MaterialDesc>& materials,
                           double dx, double dy,
                           const MeshBuildOptions& opts) {
    FemMesh mesh;
    mesh.dx = dx; mesh.dy = dy;

    // flip + convert if needed: assume input is raw BGR from imread.
    cv::Mat rgb;
    if (imageRgb.channels() == 3) {
        cv::cvtColor(imageRgb, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = imageRgb.clone();
    }
    cv::Mat flipped;
    cv::flip(rgb, flipped, 0); // y down -> y up

    mesh.nx = flipped.cols;
    mesh.ny = flipped.rows;

    auto mask = buildMask(flipped, materials);

    // material count for air: use materials.size() as air index if air not declared.
    // extract contours
    auto contours = extractContours(mask, mesh.nx, mesh.ny, dx, dy);
    // simplify each contour
    double tol = opts.boundaryTolerance > 0 ? opts.boundaryTolerance : 0.5 * dx;
    double minS = opts.minSpacing > 0 ? opts.minSpacing : dx;
    double maxS = opts.maxSpacing > 0 ? opts.maxSpacing : 10.0 * dx;
    for (auto& c : contours) {
        c.points = simplifyContour(c.points, tol, minS, maxS);
    }
    mesh.contours = contours;

    // Collect boundary vertices from contours.
    std::vector<Point2D> bndPts;
    std::vector<int> bndMat;
    for (size_t ci = 0; ci < contours.size(); ++ci) {
        for (const auto& p : contours[ci].points) {
            bndPts.push_back(p);
            bndMat.push_back(contours[ci].materialInside);
        }
    }

    // Interior points: uniform grid at maxSpacing, kept if inside any non-air material.
    // Build a set of material polygons for containment: use mask raster test instead.
    std::vector<Point2D> intPts;
    std::vector<int> intMat;
    double step = maxS;
    if (step < minS) step = minS;
    for (double y = step * 0.5; y < mesh.ny * dy; y += step) {
        for (double x = step * 0.5; x < mesh.nx * dx; x += step) {
            int ix = std::min(mesh.nx - 1, std::max(0, (int)(x / dx)));
            int iy = std::min(mesh.ny - 1, std::max(0, (int)(y / dy)));
            int mid = mask[(size_t)iy * mesh.nx + ix];
            if (mid < 0) continue; // air
            intPts.push_back({x, y});
            intMat.push_back(mid);
        }
    }

    // Assemble vertex list.
    std::vector<Point2D> allPts = bndPts;
    std::vector<int> allMat = bndMat;
    for (size_t i = 0; i < intPts.size(); ++i) {
        allPts.push_back(intPts[i]);
        allMat.push_back(intMat[i]);
    }

    // Deduplicate vertices by coordinate (CDT requires unique vertices).
    std::map<std::pair<long long,long long>, int> uniqMap;
    std::vector<Point2D> uPts;
    std::vector<int> uMat;
    std::vector<int> idxMap(allPts.size());
    for (size_t i = 0; i < allPts.size(); ++i) {
        long long kx = (long long)std::llround(allPts[i].x * 1e9);
        long long ky = (long long)std::llround(allPts[i].y * 1e9);
        auto key = std::make_pair(kx, ky);
        auto it = uniqMap.find(key);
        if (it == uniqMap.end()) {
            uniqMap[key] = (int)uPts.size();
            uPts.push_back(allPts[i]);
            uMat.push_back(allMat[i]);
            idxMap[i] = (int)uPts.size() - 1;
        } else {
            idxMap[i] = it->second;
        }
    }
    allPts = uPts;
    allMat = uMat;

    // Map each contour point to its deduplicated vertex index.
    std::vector<std::vector<int>> contourVertIdx(contours.size());
    for (size_t ci = 0; ci < contours.size(); ++ci) {
        for (const auto& p : contours[ci].points) {
            long long kx = (long long)std::llround(p.x * 1e9);
            long long ky = (long long)std::llround(p.y * 1e9);
            auto it = uniqMap.find({kx, ky});
            contourVertIdx[ci].push_back(it != uniqMap.end() ? it->second : -1);
        }
    }

    // CDT vertices.
    std::vector<CDT::V2d<double>> verts;
    verts.reserve(allPts.size());
    for (const auto& p : allPts) verts.push_back(CDT::V2d<double>(p.x, p.y));

    // constraint edges from contours using deduplicated indices.
    std::vector<CDT::Edge> edges;
    for (size_t ci = 0; ci < contours.size(); ++ci) {
        const auto& idxs = contourVertIdx[ci];
        size_t n = idxs.size();
        for (size_t i = 0; i < n; ++i) {
            int a = idxs[i];
            int b = idxs[(i + 1) % n];
            if (a < 0 || b < 0 || a == b) continue;
            edges.push_back(CDT::Edge((CDT::VertInd)a, (CDT::VertInd)b));
        }
    }

    CDT::Triangulation<double> tri(CDT::VertexInsertionOrder::Auto,
                                    CDT::IntersectingConstraintEdges::NotAllowed,
                                    0.0);
    tri.insertVertices(verts);
    try {
        tri.insertEdges(edges);
        // Ruppert-style refinement to remove small angles from staircased boundaries.
        try {
            tri.refineTriangles((CDT::VertInd)opts.maxNodes, CDT::RefinementCriterion::SmallestAngle,
                                CDT::degToRad((double)opts.minAngleDeg), nullptr, 1e-9,
                                CDT::RefinementOrder::WorstFirst);
        } catch (...) {
            // refinement failure is non-fatal for Phase 0
        }
    } catch (...) {
        // constraint insertion failure -> record and return empty mesh
        mesh.quality.passed = false;
        mesh.quality.failureReason = "CDT constraint insertion failed";
        return mesh;
    }
    tri.eraseOuterTrianglesAndHoles();

    // Build FemMesh.
    mesh.vertices.reserve(tri.vertices.size());
    for (size_t i = 0; i < tri.vertices.size(); ++i) {
        MeshVertex mv;
        mv.p = {tri.vertices[i].x, tri.vertices[i].y};
        // material: nearest original vertex material (fallback 0)
        if (i < allMat.size()) mv.material = allMat[i];
        else mv.material = 0;
        mesh.vertices.push_back(mv);
    }
    // Fix material assignment: use centroid-in-mask for each triangle instead.
    mesh.triangles.reserve(tri.triangles.size());
    for (const auto& t : tri.triangles) {
        MeshTriangle mt;
        mt.v[0] = (int)t.vertices[0];
        mt.v[1] = (int)t.vertices[1];
        mt.v[2] = (int)t.vertices[2];
        if (mt.v[0] == mt.v[1] || mt.v[0] == mt.v[2] || mt.v[1] == mt.v[2]) continue;
        double ax = mesh.vertices[mt.v[0]].p.x, ay = mesh.vertices[mt.v[0]].p.y;
        double bx = mesh.vertices[mt.v[1]].p.x, by = mesh.vertices[mt.v[1]].p.y;
        double cx = mesh.vertices[mt.v[2]].p.x, cy = mesh.vertices[mt.v[2]].p.y;
        double area = (bx-ax)*(cy-ay) - (cx-ax)*(by-ay);
        if (std::fabs(area) < 1e-18) continue;
        // centroid
        double ccx = (ax+bx+cx)/3.0;
        double ccy = (ay+by+cy)/3.0;
        int ix = std::min(mesh.nx - 1, std::max(0, (int)(ccx / dx)));
        int iy = std::min(mesh.ny - 1, std::max(0, (int)(ccy / dy)));
        int mid = mask[(size_t)iy * mesh.nx + ix];
        mt.material = (mid < 0) ? 0 : mid;
        mesh.triangles.push_back(mt);
    }

    // Quality metrics.
    double minA = 180, maxA = 0;
    int inv = 0, nonconf = 0;
    for (const auto& t : mesh.triangles) {
        std::vector<Point2D> tri = { mesh.vertices[t.v[0]].p, mesh.vertices[t.v[1]].p, mesh.vertices[t.v[2]].p };
        double a = minAngleDeg(tri);
        // also compute max angle
        auto ang = [&](int i, int j, int k) {
            double ax = tri[j].x - tri[i].x, ay = tri[j].y - tri[i].y;
            double bx = tri[k].x - tri[i].x, by = tri[k].y - tri[i].y;
            double dot = ax*bx + ay*by;
            double la = std::hypot(ax, ay), lb = std::hypot(bx, by);
            if (la < 1e-30 || lb < 1e-30) return 0.0;
            double c = dot / (la * lb);
            c = std::max(-1.0, std::min(1.0, c));
            return std::acos(c) * 180.0 / M_PI;
        };
        double a1 = ang(0,1,2), a2 = ang(1,0,2), a3 = ang(2,0,1);
        double mx = std::max({a1,a2,a3});
        if (a < minA) minA = a;
        if (mx > maxA) maxA = mx;
        // signed area
        double area = (tri[1].x-tri[0].x)*(tri[2].y-tri[0].y) - (tri[2].x-tri[0].x)*(tri[1].y-tri[0].y);
        if (area <= 0) inv++;
        // non-conforming: vertices have different materials
        int m0 = mesh.vertices[t.v[0]].material;
        if (m0 != mesh.vertices[t.v[1]].material || m0 != mesh.vertices[t.v[2]].material) nonconf++;
    }
    if (mesh.triangles.empty()) { minA = 0; maxA = 0; }
    mesh.quality.minAngleDeg = minA;
    mesh.quality.maxAngleDeg = maxA;
    mesh.quality.invertedCount = inv;
    mesh.quality.nonConformingCount = nonconf;
    mesh.quality.coverageLoss = 0.0;
    mesh.quality.passed = mesh.triangles.size() > 0 && inv == 0 && minA > 1.0;
    if (!mesh.quality.passed) {
        if (mesh.triangles.empty()) mesh.quality.failureReason = "no triangles produced";
        else if (inv > 0) mesh.quality.failureReason = "inverted triangles";
        else mesh.quality.failureReason = "min angle below gate";
    }

    return mesh;
}

std::string FemMesh::hash() const {
    std::ostringstream ss;
    ss.precision(17);
    for (const auto& v : vertices) { ss << v.p.x << "," << v.p.y << "," << v.material << ";"; }
    for (const auto& t : triangles) { ss << t.v[0] << "," << t.v[1] << "," << t.v[2] << "," << t.material << ";"; }
    std::string s = ss.str();
    std::vector<unsigned char> hash(picosha2::k_digest_size);
    picosha2::hash256(s.begin(), s.end(), hash.begin(), hash.end());
    return picosha2::bytes_to_hex_string(hash.begin(), hash.end());
}

} // namespace fem
} // namespace omfdm

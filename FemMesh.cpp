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

static double signedPolygonArea(const std::vector<Point2D>& poly) {
    double area = 0.0;
    if (poly.size() < 3) return 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];
        area += a.x * b.y - b.x * a.y;
    }
    return 0.5 * area;
}

static bool sameContourGeometry(const std::vector<Point2D>& a, const std::vector<Point2D>& b, double tol) {
    if (a.size() < 3 || b.size() < 3) return false;
    auto metrics = [](const std::vector<Point2D>& p) {
        double cx=0, cy=0, minx=1e300, maxx=-1e300, miny=1e300, maxy=-1e300;
        for (const auto& q : p) {
            cx += q.x; cy += q.y;
            minx=std::min(minx,q.x); maxx=std::max(maxx,q.x);
            miny=std::min(miny,q.y); maxy=std::max(maxy,q.y);
        }
        cx /= p.size(); cy /= p.size();
        return std::array<double,6>{cx,cy,minx,maxx,miny,maxy};
    };
    const auto ma = metrics(a), mb = metrics(b);
    if (std::hypot(ma[0]-mb[0], ma[1]-mb[1]) > 2.0*tol) return false;
    if (std::fabs((ma[3]-ma[2])-(mb[3]-mb[2])) > 3.0*tol ||
        std::fabs((ma[5]-ma[4])-(mb[5]-mb[4])) > 3.0*tol) return false;
    return true;
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
    std::vector<std::pair<size_t,size_t>> stack;
    stack.push_back({first, last});
    while (!stack.empty()) {
        const auto range = stack.back();
        stack.pop_back();
        double maxD = 0.0;
        size_t idx = 0;
        for (size_t i = range.first + 1; i < range.second; ++i) {
            const double d = perpDist(pts[i], pts[range.first], pts[range.second]);
            if (d > maxD) { maxD = d; idx = i; }
        }
        if (maxD > tol && idx != 0) {
            keep[idx] = 1;
            stack.push_back({range.first, idx});
            stack.push_back({idx, range.second});
        }
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
    // Resample each edge independently, always keeping the original vertices
    // exactly. Uniform arc-length sampling of the whole loop
    // (ceil(total/spacing) points at k/count*total) silently cuts every
    // polygon corner whenever total/spacing is not an exact integer ratio:
    // the corner's arc position then falls between two samples and the CDT
    // polygon is chamfered, leaving corner pixels outside the mesh (NaN
    // exports) and distorting material interfaces.
    std::vector<Point2D> out;
    for (size_t i = 0; i < n; ++i) {
        const Point2D& a = in[i];
        const Point2D& b = in[(i + 1) % n];
        const double segLen = std::hypot(b.x - a.x, b.y - a.y);
        if (!out.empty() &&
            std::hypot(a.x - out.back().x, a.y - out.back().y) < 1e-12)
            continue; // duplicate/degenerate joint
        out.push_back(a);
        if (segLen < 1e-30) continue;
        const int sub = std::max(1, (int)std::ceil(segLen / spacing));
        for (int k = 1; k < sub; ++k) {
            const double t = (double)k / (double)sub;
            out.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)});
        }
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
        // Pad with a 1-pixel background border so the outer boundary of a
        // full-image material is found at the image edge (0..nx, 0..ny),
        // not one pixel inset.
        cv::Mat padded;
        cv::copyMakeBorder(bin, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(padded, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        for (size_t ci = 0; ci < contours.size(); ++ci) {
            if (contours[ci].size() < 3) continue;
            Contour c;
            for (const auto& pt : contours[ci]) {
                // subtract the 1-pixel pad offset
                c.points.push_back({(pt.x - 1) * dx, (pt.y - 1) * dy});
            }
            c.materialInside = mid;
            c.materialOutside = -1;
            if (ci < hierarchy.size() && hierarchy[ci][3] != -1) c.isHole = true;
            const double signedArea = signedPolygonArea(c.points);
            if ((!c.isHole && signedArea < 0.0) || (c.isHole && signedArea > 0.0))
                std::reverse(c.points.begin(), c.points.end());
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
                           const MeshBuildOptions& opts,
                           const std::vector<Point2D>& extraInterior) {
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
    // Replace full-image outer contours with the exact domain rectangle.
    for (auto& c : contours) {
        if (c.points.empty()) continue;
        double xmin=1e9,xmax=-1e9,ymin=1e9,ymax=-1e9;
        for (const auto& p : c.points) { xmin=std::min(xmin,p.x); xmax=std::max(xmax,p.x); ymin=std::min(ymin,p.y); ymax=std::max(ymax,p.y); }
        const double eps = 1e-12;
        if (xmin < eps && ymin < eps && std::fabs(xmax - (mesh.nx-1)*dx) < 1e-9 && std::fabs(ymax - (mesh.ny-1)*dy) < 1e-9) {
            c.points = { {0.0,0.0}, {(mesh.nx-1)*dx,0.0}, {(mesh.nx-1)*dx,(mesh.ny-1)*dy}, {0.0,(mesh.ny-1)*dy} };
        }
    }
    // Simplify shared interfaces only once so adjacent regions use exactly the
    // same constrained vertices (no hanging/disconnected boundary).
    const double tol = opts.boundaryTolerance > 0 ? opts.boundaryTolerance : 0.5 * dx;
    const double minS = opts.minSpacing > 0 ? opts.minSpacing : dx;
    const double maxS = opts.maxSpacing > 0 ? opts.maxSpacing : 10.0 * dx;
    std::vector<std::vector<int>> groups;
    for (int ci = 0; ci < (int)contours.size(); ++ci) {
        int group = -1;
        for (int gi = 0; gi < (int)groups.size(); ++gi) {
            if (contours[ci].isHole != contours[groups[gi][0]].isHole &&
                sameContourGeometry(contours[ci].points, contours[groups[gi][0]].points, std::max(dx,dy))) { group = gi; break; }
        }
        if (group < 0) { groups.push_back({ci}); }
        else groups[group].push_back(ci);
    }
    for (const auto& group : groups) {
        int canonical = group[0];
        for (int ci : group) if (contours[ci].materialInside != 0) { canonical = ci; break; }
        const std::vector<Point2D> simplified = simplifyContour(contours[canonical].points, tol, minS, maxS);
        for (int ci : group) {
            contours[ci].points = simplified;
            const double area = signedPolygonArea(contours[ci].points);
            if ((!contours[ci].isHole && area < 0.0) || (contours[ci].isHole && area > 0.0))
                std::reverse(contours[ci].points.begin(), contours[ci].points.end());
        }
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
    for (double y = step * 0.5; y <= (mesh.ny - 1) * dy + 1e-12; y += step) {
        for (double x = step * 0.5; x <= (mesh.nx - 1) * dx + 1e-12; x += step) {
            int ix = std::min(mesh.nx - 1, std::max(0, (int)(x / dx)));
            int iy = std::min(mesh.ny - 1, std::max(0, (int)(y / dy)));
            int mid = mask[(size_t)iy * mesh.nx + ix];
            if (mid < 0) continue; // air
            intPts.push_back({x, y});
            intMat.push_back(mid);
        }
    }

    // Add extra interior points (from adaptive refinement).
    for (const auto& p : extraInterior) {
        int ix = std::min(mesh.nx - 1, std::max(0, (int)(p.x / dx)));
        int iy = std::min(mesh.ny - 1, std::max(0, (int)(p.y / dy)));
        int mid = mask[(size_t)iy * mesh.nx + ix];
        if (mid < 0) continue;
        intPts.push_back(p);
        intMat.push_back(mid);
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

    // Constraint edges from contours using deduplicated indices. Material 0
    // (background/air) is visited first so an interface is not inserted twice
    // with conflicting hole parity.
    std::vector<CDT::Edge> edges;
    std::set<std::pair<int,int>> constraintSet;
    for (size_t ci = 0; ci < contours.size(); ++ci) {
        const auto& idxs = contourVertIdx[ci];
        size_t n = idxs.size();
        for (size_t i = 0; i < n; ++i) {
            int a = idxs[i];
            int b = idxs[(i + 1) % n];
            if (a < 0 || b < 0 || a == b) continue;
            auto key = std::minmax(a, b);
            if (!constraintSet.insert({key.first, key.second}).second) continue;
            edges.push_back(CDT::Edge((CDT::VertInd)a, (CDT::VertInd)b));
        }
    }

    CDT::Triangulation<double> tri(CDT::VertexInsertionOrder::Auto,
                                    CDT::IntersectingConstraintEdges::NotAllowed,
                                    0.0);
    tri.insertVertices(verts);
    bool refinementFailed = false;
    try {
        tri.insertEdges(edges);
        // Ruppert-style refinement to remove small angles from staircased boundaries.
        // The refinement target must sit strictly above the acceptance gate:
        // empirically the CDT refinement stops 20-30% short of its target
        // (target 10 deg -> ~7.1 deg achieved, 15 -> ~11.9), so refining
        // exactly to the gate value guarantees spurious FEM_MESH_FAILED
        // rejections on faithful, corner-exact polygons.
        const double refineTargetDeg =
            std::min(35.0, std::max(opts.minAngleDeg * 2.0, opts.minAngleDeg + 10.0));
        try {
            tri.refineTriangles((CDT::VertInd)opts.maxNodes, CDT::RefinementCriterion::SmallestAngle,
                                CDT::degToRad(refineTargetDeg), nullptr, 1e-9,
                                CDT::RefinementOrder::WorstFirst);
        } catch (...) {
            refinementFailed = true;
        }
    } catch (...) {
        // constraint insertion failure -> record and return empty mesh
        mesh.quality.passed = false;
        mesh.quality.failureReason = "CDT constraint insertion failed";
        return mesh;
    }
    // Do not use CDT's hole-parity erasure here: every material region inside
    // the rectangular domain must be retained. Triangles outside the domain are
    // discarded by centroid bounds below.

    // De-duplicate again after CDT refinement (Steiner insertion can create
    // coincident points), then discard degenerate and out-of-domain triangles.
    {
        std::map<std::pair<long long,long long>, int> uniq;
        std::vector<int> remap(tri.vertices.size(), -1);
        std::vector<CDT::V2d<double>> newVerts;
        for (size_t i = 0; i < tri.vertices.size(); ++i) {
            const auto& p = tri.vertices[i];
            const auto key = std::make_pair(std::llround(p.x*1e9), std::llround(p.y*1e9));
            auto it = uniq.find(key);
            if (it == uniq.end()) {
                remap[i] = (int)newVerts.size();
                uniq[key] = remap[i];
                newVerts.push_back(p);
            } else remap[i] = it->second;
        }
        std::vector<CDT::Triangle> newTris;
        const double Lx = (mesh.nx-1)*dx, Ly = (mesh.ny-1)*dy, tol = 1e-9;
        for (const auto& t : tri.triangles) {
            CDT::Triangle nt = t;
            for (int k=0;k<3;++k) nt.vertices[k] = remap[t.vertices[k]];
            if (nt.vertices[0] == nt.vertices[1] || nt.vertices[1] == nt.vertices[2] || nt.vertices[2] == nt.vertices[0]) continue;
            const auto& a = newVerts[nt.vertices[0]];
            const auto& b = newVerts[nt.vertices[1]];
            const auto& c = newVerts[nt.vertices[2]];
            const double area2 = (b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y);
            if (std::fabs(area2) < 1e-24) continue;
            const double cx = (a.x+b.x+c.x)/3.0, cy = (a.y+b.y+c.y)/3.0;
            if (cx < -tol || cx > Lx+tol || cy < -tol || cy > Ly+tol) continue;
            newTris.push_back(nt);
        }
        std::vector<char> used(newVerts.size(), 0);
        for (const auto& t : newTris) for (int k=0;k<3;++k) used[t.vertices[k]] = 1;
        std::vector<int> compact(newVerts.size(), -1);
        std::vector<CDT::V2d<double>> finalVerts;
        for (size_t i=0;i<newVerts.size();++i) if (used[i]) {
            compact[i] = (int)finalVerts.size();
            finalVerts.push_back(newVerts[i]);
        }
        for (auto& t : newTris) for (int k=0;k<3;++k) t.vertices[k] = compact[t.vertices[k]];
        tri.vertices = finalVerts;
        tri.triangles = newTris;
    }

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

    // Repair small triangular gaps left by constrained refinement. A gap whose
    // three boundary edges are all unmatched is filled deterministically; larger
    // holes remain visible to the coverage/non-conformance quality gates.
    {
        auto ekey = [](int a, int b) { return std::minmax(a,b); };
        std::map<std::pair<int,int>,int> edgeCount;
        std::set<std::array<int,3>> existing;
        auto rebuild = [&]() {
            edgeCount.clear(); existing.clear();
            for (const auto& t : mesh.triangles) {
                std::array<int,3> key = {t.v[0],t.v[1],t.v[2]};
                std::sort(key.begin(), key.end());
                existing.insert(key);
                for (int k=0;k<3;++k) { auto e=ekey(t.v[k],t.v[(k+1)%3]); ++edgeCount[{e.first,e.second}]; }
            }
        };
        rebuild();
        bool added = true;
        while (added) {
            added = false;
            std::map<int,std::vector<int>> boundaryAdj;
            for (const auto& kv : edgeCount) if (kv.second == 1) {
                boundaryAdj[kv.first.first].push_back(kv.first.second);
                boundaryAdj[kv.first.second].push_back(kv.first.first);
            }
            for (const auto& kv : boundaryAdj) {
                const int v = kv.first;
                const auto& adj = kv.second;
                for (size_t i=0;i<adj.size();++i) for (size_t j=i+1;j<adj.size();++j) {
                    int a=adj[i], b=adj[j];
                    auto ab=ekey(a,b);
                    auto it=edgeCount.find({ab.first,ab.second});
                    if (it==edgeCount.end() || it->second!=1) continue;
                    std::array<int,3> key={v,a,b}; std::sort(key.begin(),key.end());
                    if (existing.count(key)) continue;
                    const auto& p0=mesh.vertices[v].p; const auto& p1=mesh.vertices[a].p; const auto& p2=mesh.vertices[b].p;
                    const double area2=(p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y);
                    if (std::fabs(area2)<1e-24) continue;
                    if (area2 < 0.0) std::swap(a,b);
                    const double cx=(p0.x+p1.x+p2.x)/3.0, cy=(p0.y+p1.y+p2.y)/3.0;
                    const double Lx=(mesh.nx-1)*dx, Ly=(mesh.ny-1)*dy;
                    if (cx<0 || cx>Lx || cy<0 || cy>Ly) continue;
                    int ix=std::min(mesh.nx-1,std::max(0,(int)(cx/dx)));
                    int iy=std::min(mesh.ny-1,std::max(0,(int)(cy/dy)));
                    int mid=mask[(size_t)iy*mesh.nx+ix];
                    MeshTriangle mt; mt.v[0]=v;mt.v[1]=a;mt.v[2]=b;mt.material=(mid<0?0:mid);
                    mesh.triangles.push_back(mt);
                    added=true;
                    rebuild();
                    break;
                }
                if (added) break;
            }
        }
    }

    // Quality metrics: configured angle gate plus real coverage loss.
    double minA = 180, maxA = 0, triangleArea = 0.0;
    int inv = 0, nonconf = 0;
    for (const auto& t : mesh.triangles) {
        std::vector<Point2D> tri = { mesh.vertices[t.v[0]].p, mesh.vertices[t.v[1]].p, mesh.vertices[t.v[2]].p };
        double a = minAngleDeg(tri);
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
        double area = (tri[1].x-tri[0].x)*(tri[2].y-tri[0].y) - (tri[2].x-tri[0].x)*(tri[1].y-tri[0].y);
        triangleArea += std::fabs(area);
        if (area <= 0) inv++;
    }
    std::map<std::pair<int,int>,int> edgeCount;
    for (const auto& t : mesh.triangles) {
        for (int k=0;k<3;++k) {
            auto e = std::minmax(t.v[k], t.v[(k+1)%3]);
            ++edgeCount[{e.first,e.second}];
        }
    }
    const double Lx = (mesh.nx-1)*dx, Ly = (mesh.ny-1)*dy, boundaryTol = 1e-9;
    auto onDomainBoundary = [&](int vi) {
        const auto& p = mesh.vertices[vi].p;
        return std::fabs(p.x)<boundaryTol || std::fabs(p.x-Lx)<boundaryTol || std::fabs(p.y)<boundaryTol || std::fabs(p.y-Ly)<boundaryTol;
    };
    for (const auto& kv : edgeCount) {
        if (kv.second == 2) continue;
        if (kv.second == 1 && onDomainBoundary(kv.first.first) && onDomainBoundary(kv.first.second)) continue;
        ++nonconf;
    }
    double expectedArea = 0.0;
    for (const auto& c : mesh.contours)
        expectedArea += c.isHole ? -std::fabs(signedPolygonArea(c.points))
                                 :  std::fabs(signedPolygonArea(c.points));
    const double coverageLoss = expectedArea > 1e-30 ? std::max(0.0, 1.0 - triangleArea / expectedArea) : 0.0;
    if (mesh.triangles.empty()) { minA = 0; maxA = 0; }
    mesh.quality.minAngleDeg = minA;
    mesh.quality.maxAngleDeg = maxA;
    mesh.quality.invertedCount = inv;
    mesh.quality.nonConformingCount = nonconf;
    mesh.quality.coverageLoss = coverageLoss;
    mesh.quality.passed = mesh.triangles.size() > 0 && inv == 0 && nonconf == 0 &&
                          minA + 0.05 >= opts.minAngleDeg && coverageLoss <= 0.001 && !refinementFailed;
    if (!mesh.quality.passed) {
        if (mesh.triangles.empty()) mesh.quality.failureReason = "no triangles produced";
        else if (inv > 0) mesh.quality.failureReason = "inverted triangles";
        else if (nonconf > 0) mesh.quality.failureReason = "non-conforming or hanging edges";
        else if (refinementFailed) mesh.quality.failureReason = "CDT refinement failed";
        else if (minA + 0.05 < opts.minAngleDeg) mesh.quality.failureReason = "min angle below configured gate";
        else mesh.quality.failureReason = "mesh coverage loss exceeds gate";
    }

    return mesh;
}

std::string FemMesh::hash() const {
    std::ostringstream ss;
    ss.precision(17);
    ss << nx << "," << ny << "," << dx << "," << dy << ";";
    for (const auto& c : contours) {
        ss << c.materialInside << "," << c.materialOutside << "," << (c.isHole ? 1 : 0) << ":";
        for (const auto& p : c.points) ss << p.x << "," << p.y << ";";
        ss << "|";
    }
    for (const auto& v : vertices) { ss << v.p.x << "," << v.p.y << "," << v.material << ";"; }
    for (const auto& t : triangles) { ss << t.v[0] << "," << t.v[1] << "," << t.v[2] << "," << t.material << ";"; }
    std::string s = ss.str();
    std::vector<unsigned char> hash(picosha2::k_digest_size);
    picosha2::hash256(s.begin(), s.end(), hash.begin(), hash.end());
    return picosha2::bytes_to_hex_string(hash.begin(), hash.end());
}

} // namespace fem
} // namespace omfdm

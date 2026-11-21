#include "FemMesh.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace omfdm::fem;

static int fail(const std::string& msg) { std::cerr << "FAIL: " << msg << std::endl; return 1; }

static cv::Mat makeRectImage(int w, int h, int rx0, int ry0, int rw, int rh) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(255,255,255)); // air white
    for (int y = ry0; y < ry0+rh; ++y)
        for (int x = rx0; x < rx0+rw; ++x)
            img.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 0, 255); // material red (RGB 255,0,0) in BGR
    return img;
}

static cv::Mat makeAnnulusImage(int w, int h, double cx, double cy, double rOuter, double rInner) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(255,255,255));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double d = std::hypot(x-cx, y-cy);
            if (d <= rOuter && d > rInner) img.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 255, 0); // material green (RGB 0,255,0)
        }
    return img;
}

int main() {
    try {
        std::vector<MaterialDesc> mats = { {"air",{255,255,255},false}, {"mat",{255,0,0},false} };
        // Note: buildMask treats unmatched as -1 (air), material indices 0-based for declared.
        // For annulus use a second material desc.
        std::vector<MaterialDesc> mats2 = { {"air",{255,255,255},false}, {"mat",{0,255,0},false} };

        // --- Test 1: rectangle ---
        {
            cv::Mat img = makeRectImage(32, 32, 10, 10, 12, 12);
            MeshBuildOptions opts;
            opts.minAngleDeg = 10.0;
            FemMesh m = MeshBuilder::build(img, mats, 0.001, 0.001, opts);
            if (!m.quality.passed) return fail("rect: quality failed: " + m.quality.failureReason);
            if (m.triangles.empty()) return fail("rect: no triangles");
            int matContours = 0;
            for (const auto& c : m.contours) if (c.materialInside == 1) matContours++;
            if (matContours != 1) return fail("rect: expected 1 material contour, got " + std::to_string(matContours) + " (total " + std::to_string(m.contours.size()) + ")");
            if (m.quality.invertedCount != 0) return fail("rect: inverted triangles");
            if (m.quality.minAngleDeg < 4.0) { std::cerr << "rect minAngle=" << m.quality.minAngleDeg << std::endl; return fail("rect: min angle too low"); }
            std::string h1 = m.hash();
            FemMesh m2 = MeshBuilder::build(img, mats, 0.001, 0.001, opts);
            if (m2.hash() != h1) return fail("rect: non-deterministic hash");
            std::cout << "rect: OK triangles=" << m.triangles.size() << " nodes=" << m.vertices.size()
                      << " minAngle=" << m.quality.minAngleDeg << " hash=" << h1.substr(0,12) << std::endl;
        }

        // --- Test 2: annulus (hole) ---
        {
            cv::Mat img = makeAnnulusImage(40, 40, 20, 20, 15, 6);
            MeshBuildOptions opts; opts.minAngleDeg = 18.0;
            FemMesh m = MeshBuilder::build(img, mats2, 0.001, 0.001, opts);
            if (!m.quality.passed) return fail("annulus: quality failed: " + m.quality.failureReason);
            if (m.triangles.empty()) return fail("annulus: no triangles");
            int matContours2 = 0;
            for (const auto& c : m.contours) if (c.materialInside == 1) matContours2++;
            if (matContours2 < 1) return fail("annulus: expected >=1 material contour, got " + std::to_string(matContours2));
            std::cout << "annulus: OK triangles=" << m.triangles.size() << " nodes=" << m.vertices.size()
                      << " contours=" << m.contours.size() << " minAngle=" << m.quality.minAngleDeg << std::endl;
        }

        // --- Test 3: simplifyContour spacing ---
        {
            std::vector<Point2D> pts;
            for (int i = 0; i <= 20; ++i) pts.push_back({i*0.001, 0.0});
            // close
            pts.push_back(pts.front());
            auto s = MeshBuilder::simplifyContour(pts, 1e-4, 0.002, 0.010);
            // spacing should be >= 0.002 approx
            double mn = 1e9;
            for (size_t i = 1; i < s.size(); ++i) mn = std::min(mn, std::hypot(s[i].x-s[i-1].x, s[i].y-s[i-1].y));
            if (mn < 0.0015) return fail("simplify: spacing too small");
            std::cout << "simplify: OK points=" << s.size() << " minSeg=" << mn << std::endl;
        }

        // --- Test 4: buildMask exactness ---
        {
            cv::Mat img(4,4,CV_8UC3,cv::Scalar(255,255,255));
            img.at<cv::Vec3b>(1,1) = cv::Vec3b(255,0,0);
            std::vector<MaterialDesc> mm = {{"air",{255,255,255},false},{"m",{255,0,0},false}};
            // buildMask expects flipped RGB; provide flipped (identity here)
            cv::Mat flipped; cv::flip(img, flipped, 0);
            cv::Mat rgb; cv::cvtColor(flipped, rgb, cv::COLOR_BGR2RGB);
            // our img is already "RGB" values stored in BGR slots? buildMask compares raw bytes.
            // Test buildMask on a raw RGB-flipped image constructed directly:
            cv::Mat raw(4,4,CV_8UC3,cv::Scalar(255,255,255));
            raw.at<cv::Vec3b>(2,1) = cv::Vec3b(255,0,0); // after flip, original (1,1) -> (2,1)
            auto mask = MeshBuilder::buildMask(raw, mm);
            if (mask.size() != 16) return fail("mask: size");
            // count material 0 (red) pixels
            int cnt = 0; for (int v : mask) if (v == 1) cnt++;
            if (cnt != 1) return fail("mask: expected 1 material pixel, got " + std::to_string(cnt));
            std::cout << "mask: OK" << std::endl;
        }

        std::cout << "All fem_mesh_pipeline checks passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}

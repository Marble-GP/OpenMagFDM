#pragma once
// Image-based 2D FEM mesh pipeline (Phase 0).
// Produces a constrained Delaunay triangulation of the material regions
// described by a colour-coded model image. See
// docs/development/IMAGE_FEM_DESIGN.md section 5.
#ifndef FEMMESH_H
#define FEMMESH_H

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace omfdm {
namespace fem {

struct Point2D {
    double x = 0.0;
    double y = 0.0;
};

struct MeshVertex {
    Point2D p;
    int material = 0;     // index into the material list
    bool boundary = false;
    int contourId = -1;
};

struct MeshTriangle {
    int v[3] = {0, 0, 0};
    int material = 0;
};

struct Contour {
    std::vector<Point2D> points; // metres, CCW outer / CW hole
    int materialInside = 0;
    int materialOutside = 0;
    bool isHole = false;
};

struct MeshQuality {
    double minAngleDeg = 0.0;
    double maxAngleDeg = 0.0;
    int invertedCount = 0;
    int nonConformingCount = 0;
    double coverageLoss = 0.0;
    bool passed = false;
    std::string failureReason;
};

struct FemMesh {
    std::vector<MeshVertex> vertices;
    std::vector<MeshTriangle> triangles;
    std::vector<Contour> contours;
    int nx = 0;
    int ny = 0;
    double dx = 0.0;
    double dy = 0.0;
    MeshQuality quality;
    std::string hash() const;
};

// Options for the mesh pipeline. All lengths are in metres.
struct MeshBuildOptions {
    double boundaryTolerance = 0.0; // 0 -> 0.5*dx
    double minSpacing = 0.0;        // 0 -> dx
    double maxSpacing = 0.0;        // 0 -> 10*dx
    double boundaryAttraction = 0.6;
    double minAngleDeg = 25.0;
    double maxGrowthRatio = 1.4;
    int maxNodes = 200000;
    bool subpixelBoundaries = false;
};

// Material descriptor used by the mask builder.
struct MaterialDesc {
    std::string name;
    std::array<uint8_t, 3> rgb = {255, 255, 255};
    bool antiAliasing = false;
};

class MeshBuilder {
public:
    // Build a per-pixel material index mask (nx*ny) from an already-flipped
    // RGB image. Unmatched pixels get material 0 (air). Exact RGB equality,
    // matching MagneticFieldAnalyzer::setupMaterialProperties.
    static std::vector<int> buildMask(const cv::Mat& imageRgb,
                                      const std::vector<MaterialDesc>& materials);

    // Extract closed contours (in metres) from a material mask.
    // Outer boundaries are CCW, holes are CW. Triple points are merged.
    static std::vector<Contour> extractContours(const std::vector<int>& mask,
                                                int nx, int ny,
                                                double dx, double dy);

    // Douglas-Peucker + arc-length resampling. Returns the simplified contour.
    static std::vector<Point2D> simplifyContour(const std::vector<Point2D>& in,
                                                double tolerance,
                                                double minSpacing,
                                                double maxSpacing);

    // Full pipeline. On failure returns a mesh with quality.passed==false and
    // quality.failureReason set; never throws for quality failures.
    static FemMesh build(const cv::Mat& imageRgb,
                         const std::vector<MaterialDesc>& materials,
                         double dx, double dy,
                         const MeshBuildOptions& opts);
};

} // namespace fem
} // namespace omfdm

#endif // FEMMESH_H

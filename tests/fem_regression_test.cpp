#include "FemEngine.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace omfdm::fem;

namespace fs = std::filesystem;

static int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

static const fs::path root = fs::temp_directory_path() / "omfdm_fem_regression";

static void writeText(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << text;
}

static cv::Mat whiteImage(int n) {
    return cv::Mat(n, n, CV_8UC3, cv::Scalar(255,255,255));
}

static std::string baseConfig(double dx, const std::string& materials,
                              const std::string& boundaries,
                              const std::string& extraSolver = "",
                              const std::string& extra = "") {
    std::ostringstream c;
    c << "coordinate_system: cartesian\n";
    c << "mesh: { dx: " << dx << ", dy: " << dx << " }\n";
    c << materials << "\n";
    c << boundaries << "\n";
    c << "solver:\n  engine: fem\n  fem:\n";
    c << "    mesh: { min_spacing: " << dx << ", max_spacing: " << (3*dx)
      << ", min_angle_deg: 10 }\n";
    c << extraSolver << "\n";
    c << "export: { format: csv, async: false }\n";
    c << extra;
    return c.str();
}

static const std::string zeroBoundaries =
    "boundary_conditions:\n"
    "  left: { type: dirichlet, value: 0.0 }\n"
    "  right: { type: dirichlet, value: 0.0 }\n"
    "  bottom: { type: dirichlet, value: 0.0 }\n"
    "  top: { type: dirichlet, value: 0.0 }\n";

static bool fileContains(const fs::path& path, const std::string& needle) {
    std::ifstream in(path);
    std::ostringstream data;
    data << in.rdbuf();
    return data.str().find(needle) != std::string::npos;
}

static double forceX(const fs::path& csv, const std::string& material) {
    std::ifstream in(csv);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind(material + ",", 0) != 0) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) cells.push_back(cell);
        if (cells.size() > 4) return std::stod(cells[4]);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

static int testNonlinear() {
    const fs::path dir = root / "nonlinear";
    fs::create_directories(dir);
    const int n = 20;
    const double dx = 0.001;
    cv::Mat img = whiteImage(n);
    for (int y = 7; y < 13; ++y)
        for (int x = 7; x < 13; ++x)
            img.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255); // RGB red after conversion
    cv::imwrite((dir / "model.png").string(), img);
    const std::string materials =
        "materials:\n"
        "  air: { rgb: [255,255,255], mu_r: 1.0, jz: 0.0 }\n"
        "  iron: { rgb: [255,0,0], mu_r: 1000.0, jz: 1.0e6, B-H: [[0,100,300,600,1000,1500,2500,5000,20000,80000],[0,0.2,0.6,1.0,1.3,1.5,1.7,1.9,2.1,2.3]] }\n";
    const std::string solver =
        "    refinement: { enabled: false }\n"
        "nonlinear_solver: { enabled: true, max_iterations: 50, tolerance: 1.0e-3, jacobian: tangent }";
    writeText(dir / "config.yaml", baseConfig(dx, materials, zeroBoundaries, solver));

    FemEngine engine((dir / "config.yaml").string(), (dir / "model.png").string());
    FemEngineResult info;
    if (engine.run((dir / "out").string(), &info) != 0)
        return fail("nonlinear: engine failed: " + info.error);
    if (!engine.converged() || engine.residual() >= 1.0e-3)
        return fail("nonlinear: residual did not converge");
    double maxMu = 0.0;
    for (double mu : engine.elementMu()) maxMu = std::max(maxMu, mu);
    const double mu0 = 4.0 * M_PI * 1e-7;
    if (maxMu < 10.0 * mu0)
        return fail("nonlinear: element Mu does not reflect B-H secant permeability");
    if (!fileContains(dir / "out" / "Mesh" / "mesh.json", "\"missing_samples\""))
        return fail("nonlinear: mesh artifact missing status fields");

    writeText(dir / "invalid.yaml",
        baseConfig(dx,
            "materials:\n  air: { rgb: [255,255,255], mu_r: 1.0 }\n"
            "  bad: { rgb: [255,0,0], B-H: [[0],[0]] }\n",
            zeroBoundaries));
    FemEngine invalid((dir / "invalid.yaml").string(), (dir / "model.png").string());
    FemEngineResult invalidInfo;
    if (invalid.run((dir / "invalid_out").string(), &invalidInfo) != 1 ||
        invalidInfo.error.find("FEM_INVALID_BH") == std::string::npos)
        return fail("nonlinear: one-point B-H table was not rejected explicitly");

    writeText(dir / "formula.yaml",
        baseConfig(dx,
            "materials:\n  air: { rgb: [255,255,255], mu_r: 1.0 }\n"
            "  formula: { rgb: [255,0,0], mu_r: \"2*pi\" }\n",
            zeroBoundaries));
    FemEngine formula((dir / "formula.yaml").string(), (dir / "model.png").string());
    FemEngineResult formulaInfo;
    if (formula.run((dir / "formula_out").string(), &formulaInfo) != 1 ||
        formulaInfo.error.find("FEM_UNSUPPORTED") == std::string::npos)
        return fail("nonlinear: formula material was not rejected explicitly");

    writeText(dir / "nonconverged.yaml",
        baseConfig(dx, materials, zeroBoundaries,
            "nonlinear_solver: { enabled: true, max_iterations: 1, tolerance: 1.0e-30, jacobian: tangent }"));
    FemEngine stalled((dir / "nonconverged.yaml").string(), (dir / "model.png").string());
    FemEngineResult stalledInfo;
    if (stalled.run((dir / "stalled_out").string(), &stalledInfo) != 2 ||
        stalledInfo.error.find("FEM_NONCONVERGED") == std::string::npos)
        return fail("nonlinear: non-convergence exit code was swallowed");

    std::cout << "nonlinear: OK residual=" << engine.residual() << " maxMu/mu0=" << maxMu/mu0 << std::endl;
    return 0;
}

static int testBoundaryConditions() {
    const fs::path dir = root / "bc";
    fs::create_directories(dir);
    const int n = 21;
    const double dx = 0.001;
    cv::imwrite((dir / "model.png").string(), whiteImage(n));
    const std::string materials = "materials:\n  air: { rgb: [255,255,255], mu_r: 1.0, jz: 0.0 }";
    const double L = (n - 1) * dx;

    std::ostringstream dirichletNeumann;
    dirichletNeumann << "boundary_conditions:\n"
        << "  left: { type: dirichlet, value: 0.0 }\n"
        << "  right: { type: dirichlet, value: " << L << " }\n"
        << "  bottom: { type: neumann }\n"
        << "  top: { type: neumann }\n";
    writeText(dir / "dirichlet.yaml", baseConfig(dx, materials, dirichletNeumann.str()));
    FemEngine linear((dir / "dirichlet.yaml").string(), (dir / "model.png").string());
    if (linear.run((dir / "dirichlet_out").string()) != 0) {
        double e = 0.0;
        for (int i = 0; i < linear.solution().size(); ++i)
            e = std::max(e, std::fabs(linear.solution()[i] - linear.mesh().vertices[i].p.x));
        return fail("bc: Dirichlet/Neumann solve failed residual=" + std::to_string(linear.residual()) + " maxErr=" + std::to_string(e));
    }
    double maxErr = 0.0;
    for (int i = 0; i < linear.solution().size(); ++i)
        maxErr = std::max(maxErr, std::fabs(linear.solution()[i] - linear.mesh().vertices[i].p.x));
    if (maxErr > 1.0e-7)
        return fail("bc: nonzero Dirichlet elimination is incorrect: " + std::to_string(maxErr));

    std::ostringstream robin;
    robin << "boundary_conditions:\n"
          << "  left: { type: dirichlet, value: 0.0 }\n"
          << "  right: { type: robin, alpha: 0.0, beta: 1.0, gamma: 1.0 }\n"
          << "  bottom: { type: neumann }\n"
          << "  top: { type: neumann }\n";
    writeText(dir / "robin.yaml", baseConfig(dx, materials, robin.str()));
    FemEngine natural((dir / "robin.yaml").string(), (dir / "model.png").string());
    if (natural.run((dir / "robin_out").string()) != 0)
        return fail("bc: Robin solve failed");
    maxErr = 0.0;
    for (int i = 0; i < natural.solution().size(); ++i)
        maxErr = std::max(maxErr, std::fabs(natural.solution()[i] - natural.mesh().vertices[i].p.x));
    if (maxErr > 2.0e-3)
        return fail("bc: Robin natural boundary is incorrect: " + std::to_string(maxErr));

    const std::string periodic =
        "boundary_conditions:\n"
        "  left: { type: periodic }\n"
        "  right: { type: periodic }\n"
        "  bottom: { type: dirichlet, value: 0.0 }\n"
        "  top: { type: dirichlet, value: 0.0 }\n";
    writeText(dir / "periodic.yaml", baseConfig(dx, materials, periodic));
    FemEngine per((dir / "periodic.yaml").string(), (dir / "model.png").string());
    const double nu = 1.0 / (4.0 * M_PI * 1e-7);
    per.setSourceFunction([&](double x, double y) {
        (void)x;
        return -nu * (M_PI*M_PI/(L*L)) * std::sin(M_PI*y/L);
    });
    if (per.run((dir / "periodic_out").string()) != 0)
        return fail("bc: periodic solve failed");
    double sum = 0.0, exactSum = 0.0, mismatch = 0.0;
    for (int i = 0; i < per.solution().size(); ++i) {
        const auto& p = per.mesh().vertices[i].p;
        const double exact = std::sin(M_PI*p.y/L);
        const double error = per.solution()[i] - exact;
        sum += error*error;
        exactSum += exact*exact;
        if (std::fabs(p.x) < 1e-12) {
            double best = 1e300;
            for (int j = 0; j < per.solution().size(); ++j) {
                const auto& q = per.mesh().vertices[j].p;
                if (std::fabs(q.x - L) < 1e-12 && std::fabs(q.y - p.y) < 1e-12) {
                    best = std::fabs(per.solution()[i] - per.solution()[j]);
                    break;
                }
            }
            mismatch = std::max(mismatch, best);
        }
    }
    if (std::sqrt(sum/exactSum) > 5.0e-2 || mismatch > 5.0e-5)
        return fail("bc: periodic constraint is inaccurate rel=" + std::to_string(std::sqrt(sum/exactSum)) + " mismatch=" + std::to_string(mismatch));

    const std::string antiperiodic =
        "boundary_conditions:\n"
        "  left: { type: periodic, value: -1 }\n"
        "  right: { type: periodic, value: -1 }\n"
        "  bottom: { type: dirichlet, value: 0.0 }\n"
        "  top: { type: dirichlet, value: 0.0 }\n";
    writeText(dir / "antiperiodic.yaml", baseConfig(dx, materials, antiperiodic));
    FemEngine anti((dir / "antiperiodic.yaml").string(), (dir / "model.png").string());
    anti.setSourceFunction([&](double x, double y) {
        const double kx = M_PI/L, ky = M_PI/L;
        return -nu * (kx*kx + ky*ky) * std::sin(kx*x) * std::sin(ky*y);
    });
    if (anti.run((dir / "antiperiodic_out").string()) != 0)
        return fail("bc: antiperiodic solve failed");
    double antiSum = 0.0, antiExactSum = 0.0, antiMismatch = 0.0;
    for (int i = 0; i < anti.solution().size(); ++i) {
        const auto& p = anti.mesh().vertices[i].p;
        const double exact = std::sin(M_PI*p.x/L) * std::sin(M_PI*p.y/L);
        const double error = anti.solution()[i] - exact;
        antiSum += error*error; antiExactSum += exact*exact;
        if (std::fabs(p.x) < 1e-12) {
            for (int j = 0; j < anti.solution().size(); ++j) {
                const auto& q = anti.mesh().vertices[j].p;
                if (std::fabs(q.x-L)<1e-12 && std::fabs(q.y-p.y)<1e-12) {
                    antiMismatch = std::max(antiMismatch, std::fabs(anti.solution()[i]+anti.solution()[j]));
                    break;
                }
            }
        }
    }
    if (std::sqrt(antiSum/antiExactSum) > 5.0e-2 || antiMismatch > 5.0e-5)
        return fail("bc: antiperiodic constraint is inaccurate rel=" + std::to_string(std::sqrt(antiSum/antiExactSum)) + " mismatch=" + std::to_string(antiMismatch));

    std::cout << "bc: OK dirichlet=" << maxErr << " periodicMismatch=" << mismatch
              << " antiperiodicMismatch=" << antiMismatch << std::endl;
    return 0;
}

static int testIntegration() {
    const fs::path dir = root / "integration";
    fs::create_directories(dir);
    const int n = 40;
    const double dx = 0.001;
    cv::Mat img = whiteImage(n);
    for (int y = 5; y < 35; ++y) {
        for (int x = 10; x < 15; ++x) img.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);
        for (int x = 25; x < 30; ++x) img.at<cv::Vec3b>(y,x) = cv::Vec3b(255,0,0);
    }
    cv::imwrite((dir / "model.png").string(), img);
    const std::string materials =
        "material_presets:\n"
        "  conductor: { mu_r: 1.0 }\n"
        "materials:\n"
        "  air: { rgb: [255,255,255], mu_r: 1.0, jz: 0.0 }\n"
        "  left: { rgb: [255,0,0], preset: conductor, jz: 1.0e6 }\n"
        "  right: { rgb: [0,0,255], preset: conductor, jz: -1.0e6 }\n";
    const std::string flux =
        "flux_linkage:\n"
        "  - { name: phi, material_a: left, material_b: right }\n";
    writeText(dir / "config.yaml", baseConfig(dx, materials, zeroBoundaries, "", flux));
    FemEngine engine((dir / "config.yaml").string(), (dir / "model.png").string());
    if (engine.run((dir / "out").string()) != 0)
        return fail("integration: force/flux solve failed");
    const double fxLeft = forceX(dir / "out" / "Forces" / "step_0001.csv", "left");
    const double fxRight = forceX(dir / "out" / "Forces" / "step_0001.csv", "right");
    if (!(fxLeft > 0.0) || !(fxRight < 0.0))
        return fail("integration: free-current force sign is reversed fx=" + std::to_string(fxLeft) + "," + std::to_string(fxRight));
    if (!fileContains(dir / "out" / "Forces" / "step_0001.csv", "Material,RGB_R,RGB_G,RGB_B,Force_X[N/m]"))
        return fail("integration: Forces CSV header does not match FDM");
    if (!fileContains(dir / "out" / "FluxLinkage" / "flux_linkage.csv", "step,phi"))
        return fail("integration: flux CSV does not use the wide step schema");
    if (!fileContains(dir / "out" / "conditions.json", "\"image_width\"") ||
        !fileContains(dir / "out" / "conditions.json", "\"engine\":\"fem\""))
        return fail("integration: conditions.json was overwritten or lost solver metadata");

    const fs::path pmDir = root / "magnetization";
    fs::create_directories(pmDir);
    cv::Mat pmImg = whiteImage(20);
    for (int y=6;y<14;++y) for(int x=6;x<14;++x) pmImg.at<cv::Vec3b>(y,x)=cv::Vec3b(0,0,255);
    cv::imwrite((pmDir / "model.png").string(), pmImg);
    writeText(pmDir / "config.yaml", baseConfig(0.001,
        "materials:\n  air: { rgb: [255,255,255], mu_r: 1.0 }\n"
        "  magnet: { rgb: [255,0,0], mu_r: 1.0, magnetization: { Br: 1.0, pattern: parallel, angle: 90 } }\n",
        zeroBoundaries));
    FemEngine pm((pmDir / "config.yaml").string(), (pmDir / "model.png").string());
    if (pm.run((pmDir / "out").string()) != 0) return fail("integration: magnetization solve failed");
    const auto m = pm.remanentMagnetization(1);
    const double mu0 = 4.0*M_PI*1e-7;
    if (std::fabs(m[0]) > 1e-8 || std::fabs(m[1] - 1.0/mu0) > 1e-6)
        return fail("integration: magnetization angle was not converted from degrees");

    const fs::path refineDir = root / "refinement";
    fs::create_directories(refineDir);
    const int rn = 16;
    const double rdx = 0.01 / (rn - 1);
    cv::imwrite((refineDir / "model.png").string(), whiteImage(rn));
    const std::string refineSolver =
        "    refinement: { enabled: true, max_rounds: 2, theta: 0.5, target_relative_error: 1.0e-8 }\n";
    writeText(refineDir / "config.yaml",
        baseConfig(rdx, "materials:\n  air: { rgb: [255,255,255], mu_r: 1.0 }", zeroBoundaries, refineSolver));
    FemEngine refined((refineDir / "config.yaml").string(), (refineDir / "model.png").string());
    const double nu = 1.0 / (4.0 * M_PI * 1e-7);
    refined.setSourceFunction([&](double x, double y) {
        return -nu * (2.0*M_PI*M_PI/(0.01*0.01)) * std::sin(M_PI*x/0.01) * std::sin(M_PI*y/0.01);
    });
    if (refined.run((refineDir / "out").string()) != 0)
        return fail("integration: adaptive refinement failed");
    if (!fileContains(refineDir / "out" / "Mesh" / "refinement.json", "\"round\":1"))
        return fail("integration: ZZ refinement did not perform a second round");

    std::cout << "integration: OK forceX=" << fxLeft << "," << fxRight << std::endl;
    return 0;
}

int main() {
    try {
        if (int rc = testNonlinear()) return rc;
        if (int rc = testBoundaryConditions()) return rc;
        if (int rc = testIntegration()) return rc;
        std::cout << "All fem_regression checks passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        return fail(std::string("exception: ") + e.what());
    }
}

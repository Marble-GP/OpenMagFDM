#pragma once
#include <Eigen/Core>
#include <string>

inline double getMuPolar(const Eigen::MatrixXd& mu_map, int r_idx, int theta_idx, const std::string& r_orientation) {
    if (r_orientation == "horizontal") {
        // mu_map shape: (ntheta, nr), indexing: (theta_idx, r_idx)
        return mu_map(theta_idx, r_idx);
    } else {
        // mu_map shape: (nr, ntheta), indexing: (r_idx, theta_idx)
        return mu_map(r_idx, theta_idx);
    }
}

inline double getJzPolar(const Eigen::MatrixXd& jz_map, int r_idx, int theta_idx, const std::string& r_orientation) {
    if (r_orientation == "horizontal") {
        // jz_map shape: (ntheta, nr), indexing: (theta_idx, r_idx)
        return jz_map(theta_idx, r_idx);
    } else {
        // jz_map shape: (nr, ntheta), indexing: (r_idx, theta_idx)
        return jz_map(r_idx, theta_idx);
    }
}

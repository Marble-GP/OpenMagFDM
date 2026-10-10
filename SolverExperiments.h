#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace omfdm {
// Research-only switches, read per solve; no process-global mutable cache and
// no public YAML/default change. Bits correspond to priorities 1 through 4.
struct SolverExperiments {
    static bool directTransientStart() {
        const char* value = std::getenv("OMFDM_NK_INITIALIZATION");
        if (!value || std::string(value) == "linear") return false;
        if (std::string(value) == "direct") return true;
        throw std::invalid_argument("OMFDM_NK_INITIALIZATION must be linear or direct");
    }
    unsigned mask = 0;
    bool materialCache() const { return mask & 1; }
    bool lazyFallback() const { return mask & 2; }
    bool krylovWorkspace() const { return mask & 4; }
    bool acceptedFields() const { return mask & 8; }
    static SolverExperiments fromEnvironment() {
        const char* value = std::getenv("OMFDM_TANGENT_OPTIMIZATIONS");
        if (!value) return {};
        const std::string text(value);
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("OMFDM_TANGENT_OPTIMIZATIONS must be an integer mask 0..15");
        const auto mask = std::stoul(text);
        if (mask > 15) throw std::invalid_argument("OMFDM_TANGENT_OPTIMIZATIONS must be 0..15");
        return {static_cast<unsigned>(mask)};
    }
};
} // namespace omfdm

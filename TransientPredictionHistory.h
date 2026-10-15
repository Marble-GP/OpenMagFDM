#pragma once
#include <algorithm>
#include <string>

namespace omfdm {
// Metadata only: vectors remain in the analyzer's existing two-state history.
// Construct per sweep/chunk so restart, exceptions and repeated analysis cannot
// carry prediction eligibility into another run. Empty keys mean unsupported.
class TransientPredictionHistory {
    int last_step_ = -1;
    int strict_count_ = 0;
    std::string key_;
public:
    bool ready(int step, const std::string& key) const {
        return !key.empty() && key == key_ && step == last_step_ + 1 && strict_count_ >= 2;
    }
    void observe(int step, bool strict, const std::string& key) {
        if (!strict || key.empty()) { strict_count_ = 0; key_.clear(); last_step_ = -1; return; }
        strict_count_ = step == last_step_ + 1 && key == key_ ? std::min(2, strict_count_ + 1) : 1;
        last_step_ = step;
        key_ = key;
    }
};
} // namespace omfdm

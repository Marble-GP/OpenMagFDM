#pragma once
#include <array>
#include <cassert>

namespace omfdm {
// Sorted, duplicate-coalescing row. A five-point operator plus the additive
// nullspace pin needs at most five unique entries. No heap allocation per row.
template<class Scalar> struct StencilRow {
    int count = 0;
    std::array<int, 8> column{};
    std::array<Scalar, 8> value{};
    Scalar rhs = 0.0;
    void add(int col, const Scalar& v) {
        int p = 0;
        while (p < count && column[p] < col) ++p;
        if (p < count && column[p] == col) { value[p] += v; return; }
        assert(count < static_cast<int>(column.size()));
        for (int q = count; q > p; --q) {
            column[q] = column[q - 1]; value[q] = value[q - 1];
        }
        column[p] = col; value[p] = v; ++count;
    }
};
} // namespace omfdm

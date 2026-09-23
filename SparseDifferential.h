#pragma once
#include <array>
#include <cassert>

namespace omfdm {
// Local forward differentiation, not a global AD tape. The two-hop five-point
// stencil has at most 13 columns (extra capacity allows coincident/tiny grids).
struct SparseDifferential {
    double value = 0;
    int count = 0;
    std::array<int, 32> column{};
    std::array<double, 32> derivative{};
    SparseDifferential(double v = 0) : value(v) {}
    void add(int col, double d) {
        int p = 0;
        while (p < count && column[p] < col) ++p;
        if (p < count && column[p] == col) { derivative[p] += d; return; }
        assert(count < 32);
        for (int q = count; q > p; --q) {
            column[q] = column[q-1]; derivative[q] = derivative[q-1];
        }
        column[p] = col; derivative[p] = d; ++count;
    }
    SparseDifferential& operator+=(const SparseDifferential& b) {
        value += b.value;
        for (int p = 0; p < b.count; ++p) add(b.column[p], b.derivative[p]);
        return *this;
    }
    SparseDifferential& operator-=(const SparseDifferential& b) {
        value -= b.value;
        for (int p = 0; p < b.count; ++p) add(b.column[p], -b.derivative[p]);
        return *this;
    }
};
inline SparseDifferential operator+(SparseDifferential a, const SparseDifferential& b) { return a += b; }
inline SparseDifferential operator-(SparseDifferential a, const SparseDifferential& b) { return a -= b; }
inline SparseDifferential operator-(SparseDifferential a) {
    a.value = -a.value;
    for (int p = 0; p < a.count; ++p) a.derivative[p] = -a.derivative[p];
    return a;
}
inline SparseDifferential operator*(const SparseDifferential& a, const SparseDifferential& b) {
    SparseDifferential r(a.value * b.value);
    for (int p = 0; p < a.count; ++p) r.add(a.column[p], a.derivative[p] * b.value);
    for (int p = 0; p < b.count; ++p) r.add(b.column[p], b.derivative[p] * a.value);
    return r;
}
inline SparseDifferential operator/(const SparseDifferential& a, const SparseDifferential& b) {
    SparseDifferential r(a.value / b.value);
    for (int p = 0; p < a.count; ++p) r.add(a.column[p], a.derivative[p] / b.value);
    for (int p = 0; p < b.count; ++p) r.add(b.column[p], -r.value * b.derivative[p] / b.value);
    return r;
}
} // namespace omfdm

#include <cmath>
#include <stdexcept>
#include <vector>

#include "Common/SelfAttention.h"
#include "Common/Types.h"
#include "doctest.h"

namespace {
Matrix makeInput(int n, int d, double base = 0.0) {
    Matrix m(n, Vector(d));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j) m[i][j] = std::sin(0.7 * (i + 1) * (j + 1) + base);
    return m;
}
} // namespace

TEST_CASE("attention output has the input shape") {
    SelfAttention attn(8, 2, 1);
    const Matrix out = attn.forward(makeInput(5, 8));
    REQUIRE(out.size() == 5);
    for (const auto& row : out) CHECK(row.size() == 8);
}

TEST_CASE("attention weights are row-stochastic, one matrix per head") {
    SelfAttention attn(8, 4, 1);
    std::vector<Matrix> weights;
    attn.forward(makeInput(6, 8), false, &weights);
    REQUIRE(weights.size() == 4);
    for (const auto& head : weights) {
        REQUIRE(head.size() == 6);
        for (const auto& row : head) {
            REQUIRE(row.size() == 6);
            double sum = 0.0;
            for (double w : row) {
                CHECK(w >= 0.0);
                sum += w;
            }
            CHECK(sum == doctest::Approx(1.0));
        }
    }
}

TEST_CASE("identical tokens receive uniform attention") {
    SelfAttention attn(4, 1, 2);
    const Matrix same(5, Vector{0.3, -0.1, 0.8, 0.5});
    std::vector<Matrix> weights;
    attn.forward(same, false, &weights);
    for (const auto& row : weights[0])
        for (double w : row) CHECK(w == doctest::Approx(0.2));
}

TEST_CASE("a single-token sequence attends fully to itself") {
    SelfAttention attn(4, 2, 2);
    std::vector<Matrix> weights;
    const Matrix out = attn.forward(makeInput(1, 4), false, &weights);
    REQUIRE(out.size() == 1);
    for (const auto& head : weights) CHECK(head[0][0] == doctest::Approx(1.0));
}

TEST_CASE("causal mask zeroes attention to future positions") {
    SelfAttention attn(8, 2, 3);
    std::vector<Matrix> weights;
    attn.forward(makeInput(5, 8), true, &weights);
    for (const auto& head : weights) {
        for (size_t i = 0; i < head.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < head[i].size(); ++j) {
                if (j > i) CHECK(head[i][j] == 0.0);
                sum += head[i][j];
            }
            CHECK(sum == doctest::Approx(1.0));
        }
        CHECK(head[0][0] == doctest::Approx(1.0));
    }
}

TEST_CASE("causal outputs do not depend on later tokens") {
    SelfAttention attn(8, 2, 3);
    Matrix a = makeInput(5, 8);
    Matrix b = a;
    for (double& x : b[4]) x += 5.0; // change only the last token

    const Matrix oa = attn.forward(a, true);
    const Matrix ob = attn.forward(b, true);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 8; ++j) CHECK(oa[i][j] == doctest::Approx(ob[i][j]));

    // Without the mask the earlier positions do change.
    const Matrix ua = attn.forward(a, false);
    const Matrix ub = attn.forward(b, false);
    bool changed = false;
    for (size_t j = 0; j < 8; ++j) changed |= std::fabs(ua[0][j] - ub[0][j]) > 1e-9;
    CHECK(changed);
}

TEST_CASE("cross-attention takes queries and memory of different lengths") {
    SelfAttention attn(8, 2, 4);
    std::vector<Matrix> weights;
    const Matrix out = attn.attend(makeInput(3, 8), makeInput(7, 8, 1.0), false, &weights);
    REQUIRE(out.size() == 3);
    CHECK(out[0].size() == 8);
    REQUIRE(weights.size() == 2);
    CHECK(weights[0].size() == 3);
    CHECK(weights[0][0].size() == 7);
}

TEST_CASE("attention is deterministic for a fixed seed") {
    SelfAttention a(8, 2, 11), b(8, 2, 11), c(8, 2, 12);
    const Matrix x = makeInput(4, 8);
    CHECK(a.forward(x) == b.forward(x));
    CHECK(a.forward(x) != c.forward(x));
}

TEST_CASE("attention is stable for large inputs") {
    SelfAttention attn(8, 2, 5);
    Matrix x = makeInput(4, 8);
    for (auto& row : x)
        for (double& v : row) v *= 1000.0; // huge scores would overflow a naive softmax
    for (const auto& row : attn.forward(x))
        for (double v : row) CHECK(std::isfinite(v));
}

TEST_CASE("attention validates its arguments") {
    CHECK_THROWS_AS(SelfAttention(7, 2), std::invalid_argument); // 7 not divisible by 2
    CHECK_THROWS_AS(SelfAttention(0, 1), std::invalid_argument);
    SelfAttention attn(4, 1);
    CHECK_THROWS_AS(attn.forward(Matrix{}), std::invalid_argument);
    CHECK_THROWS_AS(attn.forward(makeInput(2, 3)), std::invalid_argument);
}

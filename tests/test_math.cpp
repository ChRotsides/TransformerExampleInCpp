#include <cmath>
#include <limits>
#include <stdexcept>

#include "Common/LayerNorm.h"
#include "Common/Types.h"
#include "Common/VectorMath.h"
#include "Common/utils.h"
#include "doctest.h"

TEST_CASE("softmax sums to 1 and preserves ordering") {
    const Vector p = Utils::softmax({1.0, 2.0, 3.0});
    double sum = 0.0;
    for (double x : p) sum += x;
    CHECK(sum == doctest::Approx(1.0));
    CHECK(p[0] < p[1]);
    CHECK(p[1] < p[2]);
    // Known value: softmax([0, 0]) = [0.5, 0.5]
    const Vector u = Utils::softmax({0.0, 0.0});
    CHECK(u[0] == doctest::Approx(0.5));
}

TEST_CASE("softmax is numerically stable for large scores") {
    const Vector p = Utils::softmax({1000.0, 1001.0, 999.0});
    double sum = 0.0;
    for (double x : p) {
        CHECK(std::isfinite(x));
        sum += x;
    }
    CHECK(sum == doctest::Approx(1.0));
    // Shift invariance: same result as softmax([0, 1, -1]).
    const Vector q = Utils::softmax({0.0, 1.0, -1.0});
    for (size_t i = 0; i < p.size(); ++i) CHECK(p[i] == doctest::Approx(q[i]));
}

TEST_CASE("softmax gives zero probability to -infinity entries") {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    const Vector p = Utils::softmax({0.5, neg_inf, 0.5});
    CHECK(p[1] == 0.0);
    CHECK(p[0] == doctest::Approx(0.5));
    CHECK_THROWS_AS(Utils::softmax({neg_inf, neg_inf}), std::invalid_argument);
    CHECK_THROWS_AS(Utils::softmax({}), std::invalid_argument);
}

TEST_CASE("matmul, transpose and matvec") {
    const Matrix a = {{1, 2, 3}, {4, 5, 6}};
    const Matrix b = {{7, 8}, {9, 10}, {11, 12}};
    const Matrix c = VectorMath::matmul(a, b);
    REQUIRE(c.size() == 2);
    REQUIRE(c[0].size() == 2);
    CHECK(c[0][0] == 58);
    CHECK(c[0][1] == 64);
    CHECK(c[1][0] == 139);
    CHECK(c[1][1] == 154);

    const Matrix t = VectorMath::transpose(a);
    REQUIRE(t.size() == 3);
    CHECK(t[2][1] == 6);

    const Vector v = VectorMath::matvec(a, {1, 0, -1});
    CHECK(v[0] == -2);
    CHECK(v[1] == -2);

    CHECK_THROWS_AS(VectorMath::matmul(a, a), std::invalid_argument);
    CHECK_THROWS_AS(VectorMath::matvec(a, {1.0}), std::invalid_argument);
}

TEST_CASE("layer norm produces zero mean and unit variance") {
    LayerNorm ln(8);
    const Vector x = {10.0, 12.0, -3.0, 7.5, 100.0, 0.0, 4.0, -20.0};
    const Vector y = ln.normalize(x);
    REQUIRE(y.size() == x.size());

    double mean = 0.0;
    for (double v : y) mean += v;
    mean /= y.size();
    double var = 0.0;
    for (double v : y) var += (v - mean) * (v - mean);
    var /= y.size();

    CHECK(mean == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(var == doctest::Approx(1.0).epsilon(1e-4)); // slightly below 1 because of eps
}

TEST_CASE("layer norm handles constant input and wrong sizes") {
    LayerNorm ln(4);
    const Vector y = ln.normalize({3.0, 3.0, 3.0, 3.0});
    for (double v : y) CHECK(v == doctest::Approx(0.0));
    CHECK_THROWS_AS(ln.normalize(Vector{1.0, 2.0}), std::invalid_argument);
}

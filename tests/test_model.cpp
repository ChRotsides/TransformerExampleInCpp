#include <cmath>
#include <vector>

#include "Common/utils.h"
#include "Decoding/Decoder.h"
#include "Encoding/Encoder.h"
#include "Transformer.h"
#include "doctest.h"

namespace {
Matrix makeInput(int n, int d) {
    Matrix m(n, Vector(d));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j) m[i][j] = std::cos(0.3 * (i + 2) * (j + 1));
    return m;
}
} // namespace

TEST_CASE("encoder keeps (seq_len x model_dim) and normalises each token") {
    Encoder enc(2, 8, 2, 1);
    const Matrix out = enc.forward(makeInput(5, 8));
    REQUIRE(out.size() == 5);
    for (const auto& row : out) {
        REQUIRE(row.size() == 8);
        double mean = 0.0;
        for (double v : row) mean += v;
        CHECK(mean / 8.0 == doctest::Approx(0.0).epsilon(1e-9)); // output of a final LayerNorm
    }
    CHECK(Utils::allFinite(out));
}

TEST_CASE("decoder output has the target shape and ignores future target tokens") {
    Decoder dec(2, 8, 2, 1);
    const Matrix memory = makeInput(6, 8);
    Matrix t1 = makeInput(4, 8);
    Matrix t2 = t1;
    for (double& v : t2[3]) v += 3.0;

    const Matrix o1 = dec.forward(t1, memory);
    const Matrix o2 = dec.forward(t2, memory);
    REQUIRE(o1.size() == 4);
    CHECK(o1[0].size() == 8);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 8; ++j) CHECK(o1[i][j] == doctest::Approx(o2[i][j]));
}

TEST_CASE("transformer forward pass returns one distribution per target position") {
    const int vocab = 12;
    Transformer model(vocab, 16, 2, 4, 32, 42);
    const Matrix probs = model.forward({0, 1, 2, 3, 4}, {5, 6, 7});
    REQUIRE(probs.size() == 3);
    for (const auto& row : probs) {
        REQUIRE(static_cast<int>(row.size()) == vocab);
        double sum = 0.0;
        for (double p : row) {
            CHECK(p >= 0.0);
            sum += p;
        }
        CHECK(sum == doctest::Approx(1.0));
    }
}

TEST_CASE("transformer forward pass is deterministic for a fixed seed") {
    const std::vector<int> src = {0, 1, 2, 3}, tgt = {4, 5, 6};
    Transformer a(10, 16, 2, 2, 16, 42), b(10, 16, 2, 2, 16, 42), c(10, 16, 2, 2, 16, 43);
    const Matrix pa = a.forward(src, tgt);
    CHECK(pa == b.forward(src, tgt));
    CHECK(pa == a.forward(src, tgt)); // repeated calls do not mutate state
    CHECK(pa != c.forward(src, tgt));
}

TEST_CASE("transformer predictions at earlier positions ignore later target tokens") {
    Transformer model(10, 16, 2, 2, 16, 7);
    const std::vector<int> src = {0, 1, 2};
    const Matrix p1 = model.forward(src, {3, 4, 5});
    const Matrix p2 = model.forward(src, {3, 4, 9});
    for (size_t j = 0; j < p1[0].size(); ++j) {
        CHECK(p1[0][j] == doctest::Approx(p2[0][j]));
        CHECK(p1[1][j] == doctest::Approx(p2[1][j]));
    }
    CHECK(p1[2] != p2[2]);
}

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "Common/Embedding.h"
#include "Common/FeedForwardNetwork.h"
#include "Common/FinalLayer.h"
#include "Common/Tokenizer.h"
#include "Encoding/PositionalEncoding.h"
#include "doctest.h"

TEST_CASE("positional encoding matches the sinusoidal formula") {
    PositionalEncoding pe(10, 4);

    // Position 0: sin(0) = 0 on even indices, cos(0) = 1 on odd indices.
    const Vector& p0 = pe.getEncoding(0);
    CHECK(p0[0] == doctest::Approx(0.0));
    CHECK(p0[1] == doctest::Approx(1.0));
    CHECK(p0[2] == doctest::Approx(0.0));
    CHECK(p0[3] == doctest::Approx(1.0));

    // Position 1, model_dim 4: frequencies are 1 / 10000^(0/4) = 1 and 1 / 10000^(2/4) = 0.01.
    const Vector& p1 = pe.getEncoding(1);
    CHECK(p1[0] == doctest::Approx(std::sin(1.0)));
    CHECK(p1[1] == doctest::Approx(std::cos(1.0)));
    CHECK(p1[2] == doctest::Approx(std::sin(0.01)));
    CHECK(p1[3] == doctest::Approx(std::cos(0.01)));

    // Position 5, first pair.
    const Vector& p5 = pe.getEncoding(5);
    CHECK(p5[0] == doctest::Approx(std::sin(5.0)));
    CHECK(p5[1] == doctest::Approx(std::cos(5.0)));
}

TEST_CASE("positional encoding supports odd model_dim and checks bounds") {
    PositionalEncoding pe(3, 5);
    CHECK(pe.getEncoding(2).size() == 5);
    CHECK_THROWS_AS(pe.getEncoding(3), std::out_of_range);
    CHECK_THROWS_AS(pe.getEncoding(-1), std::out_of_range);
}

TEST_CASE("positional encoding apply adds to embeddings") {
    PositionalEncoding pe(8, 4);
    const Matrix emb(3, Vector(4, 1.0));
    const Matrix out = pe.apply(emb);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0].size() == 4);
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = 0; j < 4; ++j)
            CHECK(out[i][j] == doctest::Approx(1.0 + pe.getEncoding(static_cast<int>(i))[j]));

    CHECK_THROWS_AS(pe.apply(Matrix(9, Vector(4, 0.0))), std::out_of_range);
    CHECK_THROWS_AS(pe.apply(Matrix(2, Vector(3, 0.0))), std::invalid_argument);
}

TEST_CASE("tokenizer assigns stable ids and round-trips") {
    Tokenizer tok;
    const auto a = tok.tokenize("the cat sat on the mat");
    REQUIRE(a.size() == 6);
    CHECK(a[0] == a[4]); // "the" appears twice
    CHECK(tok.getVocabSize() == 5);
    CHECK(tok.detokenize(a) == "the cat sat on the mat");
    CHECK(tok.detokenize({99}) == "<unk>");
    CHECK(tok.tokenId("cat") == a[1]);
    CHECK(tok.tokenId("dog") == -1);

    const Vector oh = tok.oneHotEncode(a[1]);
    REQUIRE(oh.size() == 5);
    CHECK(oh[a[1]] == 1.0);
    CHECK_THROWS_AS(tok.oneHotEncode(5), std::out_of_range);
}

TEST_CASE("embedding is seeded, shaped correctly and bounds-checked") {
    Embedding e1(10, 6, 7), e2(10, 6, 7), e3(10, 6, 8);
    CHECK(e1.getEmbedding(3) == e2.getEmbedding(3));
    CHECK(e1.getEmbedding(3) != e3.getEmbedding(3));

    const Matrix m = e1.tokenToEmbeddings({0, 5, 9});
    REQUIRE(m.size() == 3);
    CHECK(m[0].size() == 6);
    CHECK(m[1] == e1.getEmbedding(5));
    CHECK_THROWS_AS(e1.getEmbedding(10), std::out_of_range);
    CHECK_THROWS_AS(e1.getEmbedding(-1), std::out_of_range);
}

TEST_CASE("embedding save/load round-trips") {
    const std::string path = "embedding_roundtrip_test.tmp";
    Embedding src(5, 3, 1), dst(5, 3, 2);
    src.saveEmbeddingMatrix(path);
    dst.loadEmbeddingMatrix(path);
    std::remove(path.c_str());
    for (int i = 0; i < 5; ++i) {
        const Vector a = src.getEmbedding(i), b = dst.getEmbedding(i);
        for (size_t j = 0; j < a.size(); ++j) CHECK(a[j] == doctest::Approx(b[j]));
    }
    CHECK_THROWS_AS(dst.loadEmbeddingMatrix("does_not_exist.tmp"), std::runtime_error);
}

TEST_CASE("feed-forward network keeps shape and is deterministic per seed") {
    FeedForwardNetwork f1(8, 0, 3), f2(8, 0, 3), f3(8, 0, 4);
    const Vector x = {0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8};
    const Vector y1 = f1.forward(x);
    REQUIRE(y1.size() == 8);
    CHECK(y1 == f2.forward(x));
    CHECK(y1 != f3.forward(x));
    CHECK_THROWS_AS(f1.forward(Vector{1.0}), std::invalid_argument);
}

TEST_CASE("final layer outputs a probability distribution") {
    FinalLayer fl(6, 11, 5);
    const Vector p = fl.forward(Vector{1, -1, 2, -2, 3, -3});
    REQUIRE(p.size() == 11);
    double sum = 0.0;
    for (double v : p) {
        CHECK(v >= 0.0);
        sum += v;
    }
    CHECK(sum == doctest::Approx(1.0));
    CHECK_THROWS_AS(fl.forward(Vector{1.0}), std::invalid_argument);
}

// SelfAttention.cpp
#include "Common/SelfAttention.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "Common/VectorMath.h"
#include "Common/utils.h"

namespace {
Matrix glorotUniform(int rows, int cols, std::mt19937& gen) {
    const double range = std::sqrt(6.0 / (rows + cols));
    std::uniform_real_distribution<double> dist(-range, range);
    Matrix m(rows, Vector(cols));
    for (auto& row : m)
        for (auto& x : row) x = dist(gen);
    return m;
}
} // namespace

SelfAttention::SelfAttention(int model_dim, int num_heads, unsigned seed)
    : model_dim(model_dim), num_heads(num_heads), head_dim(0) {
    if (model_dim <= 0 || num_heads <= 0 || model_dim % num_heads != 0)
        throw std::invalid_argument("SelfAttention: model_dim must be a positive multiple of num_heads");
    head_dim = model_dim / num_heads;

    std::mt19937 gen(seed);
    W_q = glorotUniform(model_dim, model_dim, gen);
    W_k = glorotUniform(model_dim, model_dim, gen);
    W_v = glorotUniform(model_dim, model_dim, gen);
    W_o = glorotUniform(model_dim, model_dim, gen);
}

Matrix SelfAttention::project(const Matrix& input, const Matrix& weights) const {
    Matrix out;
    out.reserve(input.size());
    for (const auto& row : input) {
        if (static_cast<int>(row.size()) != model_dim)
            throw std::invalid_argument("SelfAttention: token vector size does not match model_dim");
        out.push_back(VectorMath::matvec(weights, row));
    }
    return out;
}

Matrix SelfAttention::forward(const Matrix& input, bool causal,
                              std::vector<Matrix>* attention_weights) const {
    return attend(input, input, causal, attention_weights);
}

Matrix SelfAttention::attend(const Matrix& query_input, const Matrix& memory, bool causal,
                             std::vector<Matrix>* attention_weights) const {
    if (query_input.empty() || memory.empty())
        throw std::invalid_argument("SelfAttention: empty sequence");

    const Matrix Q = project(query_input, W_q); // (n_q x d)
    const Matrix K = project(memory, W_k);      // (n_k x d)
    const Matrix V = project(memory, W_v);      // (n_k x d)
    const size_t n_q = Q.size();
    const size_t n_k = K.size();
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const double neg_inf = -std::numeric_limits<double>::infinity();

    if (attention_weights) attention_weights->clear();

    Matrix concat(n_q, Vector(model_dim, 0.0));
    for (int h = 0; h < num_heads; ++h) {
        const int off = h * head_dim;
        Matrix weights(n_q, Vector(n_k, 0.0));
        for (size_t i = 0; i < n_q; ++i) {
            // Scaled dot-product scores of query i against every key.
            Vector scores(n_k);
            for (size_t j = 0; j < n_k; ++j) {
                if (causal && j > i) {
                    scores[j] = neg_inf;
                    continue;
                }
                double dot = 0.0;
                for (int c = 0; c < head_dim; ++c) dot += Q[i][off + c] * K[j][off + c];
                scores[j] = dot * scale;
            }
            weights[i] = Utils::softmax(scores);

            // Weighted sum of value vectors.
            for (size_t j = 0; j < n_k; ++j) {
                const double w = weights[i][j];
                if (w == 0.0) continue;
                for (int c = 0; c < head_dim; ++c) concat[i][off + c] += w * V[j][off + c];
            }
        }
        if (attention_weights) attention_weights->push_back(std::move(weights));
    }

    return project(concat, W_o);
}

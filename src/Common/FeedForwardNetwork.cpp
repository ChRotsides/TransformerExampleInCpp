// FeedForwardNetwork.cpp
#include "Common/FeedForwardNetwork.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "Common/VectorMath.h"

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

FeedForwardNetwork::FeedForwardNetwork(int model_dim, int hidden_dim, unsigned seed)
    : model_dim(model_dim), hidden_dim(hidden_dim > 0 ? hidden_dim : 4 * model_dim) {
    std::mt19937 gen(seed);
    W1 = glorotUniform(this->hidden_dim, model_dim, gen);
    W2 = glorotUniform(model_dim, this->hidden_dim, gen);
    b1.assign(this->hidden_dim, 0.0);
    b2.assign(model_dim, 0.0);
}

Vector FeedForwardNetwork::forward(const Vector& input) const {
    if (static_cast<int>(input.size()) != model_dim)
        throw std::invalid_argument("FeedForwardNetwork: input size does not match model_dim");
    Vector hidden = VectorMath::matvec(W1, input);
    for (int i = 0; i < hidden_dim; ++i) {
        hidden[i] += b1[i];
        if (hidden[i] < 0.0) hidden[i] = 0.0; // ReLU
    }
    Vector out = VectorMath::matvec(W2, hidden);
    for (int i = 0; i < model_dim; ++i) out[i] += b2[i];
    return out;
}

Matrix FeedForwardNetwork::forward(const Matrix& inputs) const {
    Matrix out;
    out.reserve(inputs.size());
    for (const auto& row : inputs) out.push_back(forward(row));
    return out;
}

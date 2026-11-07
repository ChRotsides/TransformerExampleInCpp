// FinalLayer.cpp
#include "Common/FinalLayer.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "Common/VectorMath.h"
#include "Common/utils.h"

FinalLayer::FinalLayer(int input_dim, int output_dim, unsigned seed)
    : input_dim(input_dim), output_dim(output_dim), W(output_dim, Vector(input_dim)), b(output_dim, 0.0) {
    std::mt19937 gen(seed);
    const double range = std::sqrt(6.0 / (input_dim + output_dim)); // Glorot uniform
    std::uniform_real_distribution<double> dist(-range, range);
    for (auto& row : W)
        for (auto& x : row) x = dist(gen);
}

Vector FinalLayer::logits(const Vector& input) const {
    if (static_cast<int>(input.size()) != input_dim)
        throw std::invalid_argument("FinalLayer: input size does not match input_dim");
    Vector z = VectorMath::matvec(W, input);
    for (int i = 0; i < output_dim; ++i) z[i] += b[i];
    return z;
}

Vector FinalLayer::forward(const Vector& input) const {
    return Utils::softmax(logits(input));
}

Matrix FinalLayer::forward(const Matrix& inputs) const {
    Matrix out;
    out.reserve(inputs.size());
    for (const auto& row : inputs) out.push_back(forward(row));
    return out;
}

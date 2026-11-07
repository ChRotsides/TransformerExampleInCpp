// LayerNorm.cpp
#include "Common/LayerNorm.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

LayerNorm::LayerNorm(int model_dim, double epsilon)
    : model_dim(model_dim), epsilon(epsilon), gamma(model_dim, 1.0), beta(model_dim, 0.0) {}

Vector LayerNorm::normalize(const Vector& input) const {
    if (static_cast<int>(input.size()) != model_dim)
        throw std::invalid_argument("LayerNorm: input size does not match model_dim");

    const double mean = std::accumulate(input.begin(), input.end(), 0.0) / input.size();
    double variance = 0.0;
    for (double v : input) variance += (v - mean) * (v - mean);
    variance /= input.size();

    const double inv_std = 1.0 / std::sqrt(variance + epsilon);
    Vector normalized(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        normalized[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
    return normalized;
}

Matrix LayerNorm::normalize(const Matrix& inputs) const {
    Matrix out;
    out.reserve(inputs.size());
    for (const auto& row : inputs) out.push_back(normalize(row));
    return out;
}

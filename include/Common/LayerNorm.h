// LayerNorm.h
#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "Common/Types.h"

// Layer normalization over the feature dimension of a single token vector:
//   y_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
class LayerNorm {
public:
    explicit LayerNorm(int model_dim, double epsilon = 1e-5);

    Vector normalize(const Vector& input) const;
    Matrix normalize(const Matrix& inputs) const; // row-wise

private:
    int model_dim;
    double epsilon;
    Vector gamma, beta; // learnable scale / shift (initialised to 1 / 0)
};

#endif // LAYERNORM_H

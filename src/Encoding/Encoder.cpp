// Encoder.cpp
#include "Encoding/Encoder.h"

Encoder::Encoder(int num_layers, int model_dim, int num_heads, unsigned seed) {
    for (int i = 0; i < num_layers; ++i) layers.emplace_back(model_dim, num_heads, seed + 10u * i);
}

Matrix Encoder::forward(const Matrix& inputs) const {
    Matrix outputs = inputs;
    for (const auto& layer : layers) outputs = layer.forward(outputs);
    return outputs;
}

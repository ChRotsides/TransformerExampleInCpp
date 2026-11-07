// Decoder.cpp
#include "Decoding/Decoder.h"

Decoder::Decoder(int num_layers, int model_dim, int num_heads, unsigned seed) {
    for (int i = 0; i < num_layers; ++i) layers.emplace_back(model_dim, num_heads, seed + 10u * i);
}

Matrix Decoder::forward(const Matrix& input, const Matrix& encoderOutput) const {
    Matrix outputs = input;
    for (const auto& layer : layers) outputs = layer.forward(outputs, encoderOutput);
    return outputs;
}

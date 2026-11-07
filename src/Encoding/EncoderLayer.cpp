// EncoderLayer.cpp
#include "Encoding/EncoderLayer.h"

#include "Common/VectorMath.h"

EncoderLayer::EncoderLayer(int model_dim, int num_heads, unsigned seed)
    : selfAttention(model_dim, num_heads, seed),
      feedForward(model_dim, 0, seed + 1),
      layerNorm1(model_dim),
      layerNorm2(model_dim) {}

Matrix EncoderLayer::forward(const Matrix& inputs) const {
    const Matrix attn = selfAttention.forward(inputs, /*causal=*/false);
    const Matrix x1 = layerNorm1.normalize(VectorMath::add(inputs, attn));
    const Matrix ff = feedForward.forward(x1);
    return layerNorm2.normalize(VectorMath::add(x1, ff));
}

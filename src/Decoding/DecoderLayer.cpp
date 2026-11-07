// DecoderLayer.cpp
#include "Decoding/DecoderLayer.h"

#include "Common/VectorMath.h"

DecoderLayer::DecoderLayer(int model_dim, int num_heads, unsigned seed)
    : selfAttention(model_dim, num_heads, seed),
      encDecAttention(model_dim, num_heads, seed + 1),
      feedForward(model_dim, 0, seed + 2),
      layerNorm1(model_dim),
      layerNorm2(model_dim),
      layerNorm3(model_dim) {}

Matrix DecoderLayer::forward(const Matrix& decoderInput, const Matrix& encoderOutput) const {
    // Masked self-attention: a position cannot look at later positions.
    const Matrix selfAttn = selfAttention.forward(decoderInput, /*causal=*/true);
    const Matrix x1 = layerNorm1.normalize(VectorMath::add(decoderInput, selfAttn));

    // Cross-attention: queries from the decoder, keys/values from the encoder.
    const Matrix cross = encDecAttention.attend(x1, encoderOutput, /*causal=*/false);
    const Matrix x2 = layerNorm2.normalize(VectorMath::add(x1, cross));

    const Matrix ff = feedForward.forward(x2);
    return layerNorm3.normalize(VectorMath::add(x2, ff));
}

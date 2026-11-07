// DecoderLayer.h
#ifndef DECODERLAYER_H
#define DECODERLAYER_H

#include "Common/FeedForwardNetwork.h"
#include "Common/LayerNorm.h"
#include "Common/SelfAttention.h"
#include "Common/Types.h"

// Post-norm decoder block:
//   x = LayerNorm(x + MaskedSelfAttention(x))
//   x = LayerNorm(x + CrossAttention(x, encoder_output))
//   x = LayerNorm(x + FeedForward(x))
class DecoderLayer {
public:
    DecoderLayer(int model_dim, int num_heads = 1, unsigned seed = 42);

    Matrix forward(const Matrix& decoderInput, const Matrix& encoderOutput) const;

private:
    SelfAttention selfAttention;
    SelfAttention encDecAttention; // keys/values come from the encoder output
    FeedForwardNetwork feedForward;
    LayerNorm layerNorm1, layerNorm2, layerNorm3;
};

#endif // DECODERLAYER_H

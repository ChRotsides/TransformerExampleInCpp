// EncoderLayer.h
#ifndef ENCODERLAYER_H
#define ENCODERLAYER_H

#include "Common/FeedForwardNetwork.h"
#include "Common/LayerNorm.h"
#include "Common/SelfAttention.h"
#include "Common/Types.h"

// Post-norm encoder block:
//   x = LayerNorm(x + SelfAttention(x))
//   x = LayerNorm(x + FeedForward(x))
class EncoderLayer {
public:
    EncoderLayer(int model_dim, int num_heads = 1, unsigned seed = 42);

    Matrix forward(const Matrix& inputs) const;

private:
    SelfAttention selfAttention;
    FeedForwardNetwork feedForward;
    LayerNorm layerNorm1, layerNorm2;
};

#endif // ENCODERLAYER_H

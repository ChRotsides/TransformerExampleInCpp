// Encoder.h
#ifndef ENCODER_H
#define ENCODER_H

#include <vector>

#include "Encoding/EncoderLayer.h"

// A stack of identical-shaped (independently initialised) encoder layers.
class Encoder {
public:
    Encoder(int num_layers, int model_dim, int num_heads = 1, unsigned seed = 42);

    Matrix forward(const Matrix& inputs) const;

private:
    std::vector<EncoderLayer> layers;
};

#endif // ENCODER_H

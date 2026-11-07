// Decoder.h
#ifndef DECODER_H
#define DECODER_H

#include <vector>

#include "Decoding/DecoderLayer.h"

class Decoder {
public:
    Decoder(int num_layers, int model_dim, int num_heads = 1, unsigned seed = 1000);

    Matrix forward(const Matrix& input, const Matrix& encoderOutput) const;

private:
    std::vector<DecoderLayer> layers;
};

#endif // DECODER_H

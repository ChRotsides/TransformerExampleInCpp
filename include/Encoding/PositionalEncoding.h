// PositionalEncoding.h
#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "Common/Types.h"

// Sinusoidal positional encoding (Vaswani et al., 2017):
//   PE(pos, 2k)   = sin(pos / 10000^(2k / model_dim))
//   PE(pos, 2k+1) = cos(pos / 10000^(2k / model_dim))
class PositionalEncoding {
public:
    PositionalEncoding(int max_seq_len, int model_dim);

    const Vector& getEncoding(int pos) const;

    // Returns embeddings + positional encodings (same shape as the input).
    // Throws if the sequence is longer than max_seq_len or the width differs from model_dim.
    Matrix apply(const Matrix& embeddings) const;

private:
    int max_seq_len;
    int model_dim;
    Matrix encoding_matrix;
};

#endif // POSITIONAL_ENCODING_H

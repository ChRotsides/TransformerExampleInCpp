// FinalLayer.h
#ifndef FINAL_LAYER_H
#define FINAL_LAYER_H

#include "Common/Types.h"

// Output projection: logits = W x + b, probabilities = softmax(logits).
class FinalLayer {
public:
    FinalLayer(int input_dim, int output_dim, unsigned seed = 42);

    Vector logits(const Vector& input) const;
    Vector forward(const Vector& input) const; // softmax probabilities over output_dim classes
    Matrix forward(const Matrix& inputs) const; // row-wise

private:
    int input_dim, output_dim;
    Matrix W; // (output_dim x input_dim)
    Vector b; // (output_dim)
};

#endif // FINAL_LAYER_H

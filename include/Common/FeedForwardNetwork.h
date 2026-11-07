// FeedForwardNetwork.h
#ifndef FEED_FORWARD_NETWORK_H
#define FEED_FORWARD_NETWORK_H

#include "Common/Types.h"

// Position-wise feed-forward network: Linear(d -> hidden) -> ReLU -> Linear(hidden -> d).
class FeedForwardNetwork {
public:
    // hidden_dim <= 0 selects the conventional 4 * model_dim.
    explicit FeedForwardNetwork(int model_dim, int hidden_dim = 0, unsigned seed = 42);

    Vector forward(const Vector& input) const;
    Matrix forward(const Matrix& inputs) const; // applied independently to every row

private:
    int model_dim, hidden_dim;
    Matrix W1, W2; // (hidden x d), (d x hidden)
    Vector b1, b2;
};

#endif // FEED_FORWARD_NETWORK_H

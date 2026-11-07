// Types.h
#ifndef TYPES_H
#define TYPES_H

#include <vector>

// Row-major dense containers used throughout the project.
// A sequence of token vectors is a Matrix of shape (seq_len x model_dim).
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

#endif // TYPES_H

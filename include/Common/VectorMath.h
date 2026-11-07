// VectorMath.h
#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include "Common/Types.h"

class VectorMath {
public:
    // (m x k) * (k x n) -> (m x n). Throws std::invalid_argument on shape mismatch.
    static Matrix matmul(const Matrix& a, const Matrix& b);

    // (m x n) -> (n x m)
    static Matrix transpose(const Matrix& matrix);

    // (m x n) * (n) -> (m)
    static Vector matvec(const Matrix& matrix, const Vector& vec);

    // Element-wise a + b for equally sized vectors / matrices.
    static Vector add(const Vector& a, const Vector& b);
    static Matrix add(const Matrix& a, const Matrix& b);
};

#endif // VECTOR_MATH_H

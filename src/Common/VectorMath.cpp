// VectorMath.cpp
#include "Common/VectorMath.h"

#include <stdexcept>

Matrix VectorMath::transpose(const Matrix& matrix) {
    if (matrix.empty()) return {};
    Matrix result(matrix[0].size(), Vector(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        if (matrix[i].size() != matrix[0].size())
            throw std::invalid_argument("transpose: ragged matrix");
        for (size_t j = 0; j < matrix[i].size(); ++j) result[j][i] = matrix[i][j];
    }
    return result;
}

Matrix VectorMath::matmul(const Matrix& a, const Matrix& b) {
    if (a.empty() || b.empty()) return {};
    const size_t k = b.size();
    const size_t n = b[0].size();
    Matrix result(a.size(), Vector(n, 0.0));
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != k) throw std::invalid_argument("matmul: inner dimensions do not match");
        for (size_t p = 0; p < k; ++p) {
            const double aip = a[i][p];
            for (size_t j = 0; j < n; ++j) result[i][j] += aip * b[p][j];
        }
    }
    return result;
}

Vector VectorMath::matvec(const Matrix& matrix, const Vector& vec) {
    Vector result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        if (matrix[i].size() != vec.size())
            throw std::invalid_argument("matvec: dimension mismatch");
        for (size_t j = 0; j < vec.size(); ++j) result[i] += matrix[i][j] * vec[j];
    }
    return result;
}

Vector VectorMath::add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) throw std::invalid_argument("add: size mismatch");
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

Matrix VectorMath::add(const Matrix& a, const Matrix& b) {
    if (a.size() != b.size()) throw std::invalid_argument("add: row count mismatch");
    Matrix result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = add(a[i], b[i]);
    return result;
}

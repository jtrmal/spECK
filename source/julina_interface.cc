#include <cuda_runtime_api.h>

#include "dCSR.h"
#include "Multiply.h"
#include "semiring.h"

template<typename T>
void multiply_helper(const dCSR<T> &A, const dCSR<T> &B, dCSR<T> &matOut) {
    spECK::spECKConfig config = spECK::spECKConfig::initialize(0);
    Timings timings;
    spECK::MultiplyspECK<T, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(A, B, matOut, config, timings);
}

extern "C" {

  void julia_multiply_float(size_t mA, size_t nA, size_t nB,
      size_t nnzA,   unsigned int *rowsA,    unsigned int *colsA,   float *valsA,
      size_t nnzB,   unsigned int *rowsB,    unsigned int *colsB,   float *valsB,
      size_t *nnzO,  unsigned int **rowsO,   unsigned int **colsO,  float **vals) {
    dCSR<float> A, B, matOut;

    A.rows = mA;
    A.cols = nA;
    A.nnz = nnzA;
    A.data = valsA;
    A.row_offsets = rowsA;
    A.col_ids = colsA;

    B.rows = nA;
    B.cols = nB;
    B.nnz = nnzB;
    B.data = valsB;
    B.row_offsets = rowsB;
    B.col_ids = colsB;

    multiply_helper(A, B, matOut);

    A.data = nullptr;
    A.row_offsets = nullptr;
    A.col_ids = nullptr;

    B.data = nullptr;
    B.row_offsets = nullptr;
    B.col_ids = nullptr;

    *nnzO = matOut.nnz;
    *rowsO = matOut.row_offsets;
    *colsO = matOut.col_ids;
    *vals = matOut.data;
    matOut.data = nullptr;
    matOut.row_offsets = nullptr;
    matOut.col_ids = nullptr;
  }

  void julia_multiply_realsemiring(size_t mA, size_t nA, size_t nB,
      size_t nnzA,   unsigned int *rowsA,    unsigned int *colsA,   float *valsA,
      size_t nnzB,   unsigned int *rowsB,    unsigned int *colsB,   float *valsB,
      size_t *nnzO,  unsigned int **rowsO,   unsigned int **colsO,  float **vals) {
    dCSR<Semiring> A, B, matOut;

    A.rows = mA;
    A.cols = nA;
    A.nnz = nnzA;
    A.data = (Semiring *)valsA;
    A.row_offsets = rowsA;
    A.col_ids = colsA;

    B.rows = nA;
    B.cols = nB;
    B.nnz = nnzB;
    B.data = (Semiring *)valsB;
    B.row_offsets = rowsB;
    B.col_ids = colsB;

    multiply_helper(A, B, matOut);

    A.data = nullptr;
    A.row_offsets = nullptr;
    A.col_ids = nullptr;

    B.data = nullptr;
    B.row_offsets = nullptr;
    B.col_ids = nullptr;

    *nnzO = matOut.nnz;
    *rowsO = matOut.row_offsets;
    *colsO = matOut.col_ids;
    *vals = (float *)matOut.data;
    matOut.data = nullptr;
    matOut.row_offsets = nullptr;
    matOut.col_ids = nullptr;
  }
}


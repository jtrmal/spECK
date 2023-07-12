#include <cuda_runtime_api.h>

#include "CSR.h"
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
  void julia_float_free_cpumem(unsigned int *rowsA,    unsigned int *colsA,   float *valsA) {
    CSR<float> A;
    A.rows = 10;
    A.cols = 10;
    A.nnz = 10;
    A.data = std::unique_ptr<float []>(valsA);
    A.row_offsets = std::unique_ptr<unsigned int []>(rowsA);
    A.col_ids = std::unique_ptr<unsigned int []>(colsA);
  }

  void julia_multiply_float_cpu(size_t mA, size_t nA, size_t nB,
      size_t nnzA,   unsigned int *rowsA,    unsigned int *colsA,   float *valsA,
      size_t nnzB,   unsigned int *rowsB,    unsigned int *colsB,   float *valsB,
      size_t *nnzO,  unsigned int **rowsO,   unsigned int **colsO,  float **vals) {

    CSR<float> A, B, matOut;

    A.rows = mA;
    A.cols = nA;
    A.nnz = nnzA;
    A.data = std::unique_ptr<float []>(valsA);
    A.row_offsets = std::unique_ptr<unsigned int []>(rowsA);
    A.col_ids = std::unique_ptr<unsigned int []>(colsA);

    B.rows = nA;
    B.cols = nB;
    B.nnz = nnzB;
    B.data = std::unique_ptr<float[]>(valsB);
    B.row_offsets = std::unique_ptr<unsigned int[]>(rowsB);
    B.col_ids = std::unique_ptr<unsigned int[]>(colsB);

    dCSR<float> cuA, cuB, cuMatOut;
    convert(cuA, A);
    convert(cuB, B);
    multiply_helper(cuA, cuB, cuMatOut);
    convert(matOut, cuMatOut);

    A.data.release();
    A.row_offsets.release();
    A.col_ids.release();

    B.data.release();
    B.row_offsets.release();
    B.col_ids.release();

    *nnzO = matOut.nnz;
    *rowsO = matOut.row_offsets.release();
    *colsO = matOut.col_ids.release();
    *vals = matOut.data.release();
  }


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


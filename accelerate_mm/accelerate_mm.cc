#include <time.h>
#include <cstdio>
#include <FbgemmSparse.h>
#include <Utils.h>
#include <spmmUtils.h>
#include <iostream>
#include <iomanip>
using namespace std;
using namespace fbgemm;
extern "C"{
    float* accelerate_fbgemm_matmul(int, int, int, float*, float*, float*);
}
void Print_Matrix(int row, int col, float* matrix){
  // for debug
  for(int i = 0;i < row;i++){
        for(int j = 0;j < col;j++){
            printf("%f ", *(matrix + col * i + j));
      }
      printf("\n");
  }
  printf("\n");
  return;
}

float* accelerate_fbgemm_matmul(int m, int n, int k, float* A, float* B, float* C) {
  // C is M x N
  // A is M x K
  // B is K x N

  // transpose_matrix(k, n, B, n, B, k);

  unique_ptr<CSRMatrix<float>> csr = fbgemmDenseToCSR(m, k, A);
  SparseDenseMM(
    m, 
    n, 
    csr->rowPtr.data(), 
    csr->colIdx.data(), 
    csr->values.data(), 
    B, 
    n, 
    C, 
    n);
  return C;
}

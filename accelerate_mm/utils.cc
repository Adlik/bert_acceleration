#include <time.h>
#include <cstdio>
#include <FbgemmSparse.h>
#include <Utils.h>
#include <spmmUtils.h>
#include <iostream>
#include <iomanip>

template <typename T>
void transpose_matrix(
    int M,
    int N,
    const T* src,
    int ld_src,
    T* dst,
    int ld_dst) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      dst[i * ld_dst + j] = src[i + j * ld_src];
    }
  } // for each output row
}

template <typename T>
void transpose_matrix(T* ref, int n, int k) {
  std::vector<T> local(n * k);
  transpose_matrix(n, k, ref, k, local.data(), n);
  memcpy(ref, local.data(), n * k * sizeof(T));
}
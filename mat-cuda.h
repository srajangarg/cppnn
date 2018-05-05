#pragma once

#ifdef CUDA

#include <math.h>
#include <cuda.h>

__global__ void mat_mul_device(float *a, float *b, float *out, int m, int k, int n,
                               bool trans_a, bool trans_b)
{
    extern __shared__ float temp[];

    int r = blockIdx.x / n;
    int c = blockIdx.x % n;

    int a_index = (trans_a) ? (r + threadIdx.x * m) : (r * k + threadIdx.x);
    int b_index = (trans_b) ? (c * k + threadIdx.x) : (c + threadIdx.x * n);

    temp[threadIdx.x] = a[a_index] * b[b_index];

    __syncthreads();

    if (threadIdx.x == 0) {
        out[r * n + c] = 0;
        for (int i = 0; i < blockDim.x; i++) {
            out[r * n + c] += temp[i];
        }
    }
}

void mat_mul(Tensor&a_t, Tensor&b_t, Tensor&out_t, int m, int k, int n, bool trans_a = false,
             bool trans_b = false)
{
    assert(a_t.is_cuda);
    assert(b_t.is_cuda);
    assert(out_t.is_cuda);

    float *a = a_t.data, *b = b_t.data, *out = out_t.data;

    int numBlocks, numThreads;

    numBlocks = m * n;
    numThreads = k;

    mat_mul_device<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(
        a, b, out, m, k, n, trans_a, trans_b);
}

#endif

#pragma once

#ifdef CUDA

#include <math.h>
#include <cuda.h>

__global__ void mat_mul_device(float *a, float *b, float *out, int *m_ptr, int *k_ptr, int *n_ptr,
                               bool *trans_a, bool *trans_b)
{
    extern __shared__ float temp[];

    int m = *m_ptr;
    int k = *k_ptr;
    int n = *n_ptr;
    int r = blockIdx.x / n;
    int c = blockIdx.x % m;

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

void mat_mul(float *a, float *b, float *out, int m, int k, int n, bool trans_a = false,
             bool trans_b = false)
{
    int numBlocks, numThreads;
    float *dev_a, *dev_b, *dev_out;
    int *dev_m, *dev_k, *dev_n;
    bool *dev_trans_a, *dev_trans_b;

    cudaMalloc((void **)&dev_a, m * k * sizeof(float));
    cudaMalloc((void **)&dev_b, k * n * sizeof(float));
    cudaMalloc((void **)&dev_out, m * n * sizeof(float));
    cudaMalloc((void **)&dev_m, sizeof(int));
    cudaMalloc((void **)&dev_k, sizeof(int));
    cudaMalloc((void **)&dev_n, sizeof(int));
    cudaMalloc((void **)&dev_trans_a, sizeof(bool));
    cudaMalloc((void **)&dev_trans_b, sizeof(bool));

    numBlocks = m * n;
    numThreads = k;

    cudaMemcpy(dev_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_m, &m, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_trans_a, &trans_a, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_trans_b, &trans_b, sizeof(bool), cudaMemcpyHostToDevice);

    mat_mul_device<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(
        dev_a, dev_b, dev_out, dev_m, dev_k, dev_n, dev_trans_a, dev_trans_b);

    cudaMemcpy(out, dev_out, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_m);
    cudaFree(dev_k);
    cudaFree(dev_n);
    cudaFree(dev_trans_a);
    cudaFree(dev_trans_b);
}

#endif

#pragma once

#ifdef CUDA

#include "misc.h"

__device__ __host__ inline float &at(float *f, int I, int i)
{
    return f[i];
}

__device__ __host__ inline float &at(float *f, int I, int J, int i, int j)
{
    return f[J * i + j];
}

__device__ __host__ inline float &at(float *f, int I, int J, int K, int i, int j, int k)
{
    return f[K * (J * i + j) + k];
}

__device__ __host__ inline float &at(float *f, int I, int J, int K, int L, int i, int j, int k,
                                     int l)
{
    return f[L * (K * (J * i + j) + k) + l];
}

__global__ void conv2d_device(float *img, int inF, int H, int W, float *kernel, int outF, int kH,
                              int kW, float *out, int outH, int outW, int padH, int padW)
{
    extern __shared__ float temp[];

    int of = blockIdx.x / (outH * outW);
    int i = (blockIdx.x % (outH * outW)) / outW;
    int j = blockIdx.x % outW;

    int ki = threadIdx.x / (kW * inF);
    int kj = (threadIdx.x % (kW * inF)) / inF;
    int ff = threadIdx.x % inF;

    int ii = i - padH + ki;
    int jj = j - padW + kj;

    if (ii >= 0 and ii < H and jj >= 0 and jj < W) {
        temp[threadIdx.x]
            = at(img, H, W, inF, ii, jj, ff) * at(kernel, outF, kH, kW, inF, of, ki, kj, ff);
    } else {
        temp[threadIdx.x] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float &res = at(out, H, W, outF, i, j, of);
        res = 0;
        for (int i = 0; i < blockDim.x; i++) {
            res += temp[i];
        }
    }
}

void func_conv2d(float *img, int inF, int H, int W, float *kernel, int outF, int kH, int kW,
                 float *&out, int padH = 0, int padW = 0)
{
    // img: H, W, inF
    // kernel: outF, kH, kW, inF
    // output: H, W, outF (if pad=true)
    // output: H-kH+1, W-kW+1, outF (if pad=false)

    float *dev_img, *dev_kernel, *dev_out;

    int outH = H - kH + 1 + 2 * padH;
    int outW = W - kW + 1 + 2 * padW;
    alloc_vec(out, outH * outW * outF);

    int numBlocks = outF * outH * outW;
    int numThreads = kH * kW * outF;

    cudaMalloc((void **)&dev_img, inF * H * W * sizeof(float));
    cudaMalloc((void **)&dev_kernel, inF * kH * kW * outF * sizeof(float));
    cudaMalloc((void **)&dev_out, outF * outH * outW * sizeof(float));

    cudaMemcpy(dev_img, img, inF * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel, kernel, inF * kH * kW * outF * sizeof(float), cudaMemcpyHostToDevice);

    conv2d_device<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(
        dev_img, inF, H, W, dev_kernel, outF, kH, kW, dev_out, outH, outW, padH, padW);

    cudaMemcpy(out, dev_out, outF * outH * outW * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_img);
    cudaFree(dev_kernel);
    cudaFree(dev_out);
}

#endif

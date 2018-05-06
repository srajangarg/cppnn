#pragma once

#include "misc.h"
#include "tensor.h"
#include <cuda.h>

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

void func_conv2d_cpu(Tensor &img_t, int inF, int H, int W, Tensor &kernel_t, int outF, int kH,
                     int kW, Tensor &out_t, int padH = 0, int padW = 0)
{
    // img: H, W, inF
    // kernel: outF, kH, kW, inF
    // output: H-kH+1+2*padH, W-kW+1+2*padW, outF (if pad=false)

    assert(!img_t.is_cuda);
    assert(!kernel_t.is_cuda);
    assert(!out_t.is_cuda);

    float *img = img_t.data;
    float *kernel = kernel_t.data;
    float *out = out_t.data;

    int kH_centre = kH / 2;
    int kW_centre = kW / 2;

    int outH = H - kH + 1 + 2 * padH;
    int outW = W - kW + 1 + 2 * padW;
    assert(out_t.is_shape(outH, outW, outF));
    // alloc_vec(out, outH * outW * outF);

    for (int of = 0; of < outF; ++of) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                float res = 0;
                for (int ki = 0; ki < kH; ++ki) {
                    for (int kj = 0; kj < kW; ++kj) {
                        int ii = i - padH + ki;
                        int jj = j - padW + kj;
                        if (ii >= 0 and ii < H and jj >= 0 and jj < W) {
                            for (int ff = 0; ff < inF; ++ff) {
                                res += at(img, H, W, inF, ii, jj, ff)
                                       * at(kernel, outF, kH, kW, inF, of, ki, kj, ff);
                            }
                        }
                    }
                }
                at(out, outH, outW, outF, i, j, of) = res;
            }
        }
    }
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

void func_conv2d_cuda(Tensor &img_t, int inF, int H, int W, Tensor &kernel_t, int outF, int kH,
                      int kW, Tensor &out_t, int padH = 0, int padW = 0)
{
    assert(img_t.is_cuda);
    assert(kernel_t.is_cuda);
    assert(out_t.is_cuda);

    float *dev_img = img_t.data;
    float *dev_kernel = kernel_t.data;
    float *dev_out = out_t.data;

    int outH = H - kH + 1 + 2 * padH;
    int outW = W - kW + 1 + 2 * padW;
    assert(out_t.is_shape(outH, outW, outF));

    int numBlocks = outF * outH * outW;
    int numThreads = kH * kW * outF;

    conv2d_device<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(
        dev_img, inF, H, W, dev_kernel, outF, kH, kW, dev_out, outH, outW, padH, padW);
}

void func_conv2d(Tensor &img_t, int inF, int H, int W, Tensor &kernel_t, int outF, int kH, int kW,
                 Tensor &out_t, int padH = 0, int padW = 0)
{
    assert(img_t.is_cuda == kernel_t.is_cuda);
    assert(img_t.is_cuda == out_t.is_cuda);

    if (img_t.is_cuda)
        func_conv2d_cuda(img_t, inF, H, W, kernel_t, outF, kH, kW, out_t, padH, padW);
    else
        func_conv2d_cpu(img_t, inF, H, W, kernel_t, outF, kH, kW, out_t, padH, padW);
}

#pragma once

#ifdef CUDA

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
                              int kW, float *out)
{
    extern __shared__ float temp[];

    int of = blockIdx.x / (H * W);
    int i = (blockIdx.x % (H * W)) / W;
    int j = blockIdx.x % W;

    int ki = threadIdx.x / (kW * inF);
    int kj = (threadIdx.x % (kW * inF)) / inF;
    int ff = threadIdx.x % inF;

    int ii = i - kH / 2 + ki;
    int jj = j - kW / 2 + kj;

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
                 float *out)
{
    // img: H, W, inF
    // kernel: outF, kH, kW, inF
    // output: H, W, outF (if pad=true)
    // output: H-kH+1, W-kW+1, outF (if pad=false)

    float *dev_img, *dev_kernel, *dev_out;

    int numBlocks = outF * H * W;
    int numThreads = kH * kW * outF;

    cudaMalloc((void **)&dev_img, inF * H * W * sizeof(float));
    cudaMalloc((void **)&dev_kernel, inF * kH * kW * outF * sizeof(float));
    cudaMalloc((void **)&dev_out, outF * H * W * sizeof(float));

    cudaMemcpy(dev_img, img, inF * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel, kernel, inF * kH * kW * outF * sizeof(float), cudaMemcpyHostToDevice);

    conv2d_device<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(
        dev_img, inF, H, W, dev_kernel, outF, kH, kW, dev_out);

    cudaMemcpy(out, dev_out, outF * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_img);
    cudaFree(dev_kernel);
    cudaFree(dev_out);
}

#endif

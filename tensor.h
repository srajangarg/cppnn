#pragma once

#define NDIM_MAX 4

#include <stdarg.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <cassert>
#include <omp.h>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#define DELETE_NULL(x)                                                                             \
    if (x != NULL) {                                                                               \
        delete x;                                                                                  \
        x = NULL;                                                                                  \
    }
#define DELETE_CUDA_NULL(x)                                                                        \
    if (x != NULL) {                                                                               \
        cudaFree(x);                                                                               \
        x = NULL;                                                                                  \
    }
#define DELETE_VEC_NULL(x)                                                                         \
    if (x != NULL) {                                                                               \
        delete[] x;                                                                                \
        x = NULL;                                                                                  \
    }

__global__ void tensor_add_(float *lhs, float *rhs, int size, float scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        lhs[index] += scale * rhs[index];
}
__global__ void tensor_add_(float *lhs, int size, float val)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        lhs[index] += val;
}
__global__ void tensor_sub(float *out, float *first, float *second, int size, float scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        out[index] = scale * (first[index] - second[index]);
}
__global__ void tensor_mul_(float *tensor, int size, float val)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        tensor[index] *= val;
}
__global__ void tensor_fill_(float *tensor, int size, float val)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        tensor[index] = val;
}
__global__ void tensor_clip_(float *lhs, int size, float start, float end)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        lhs[index] = min(max(lhs[index], start), end);
}
__global__ void tensor_mag_components_(float *lhs, float *rhs, int size, float scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        lhs[index] = sqrt(pow(lhs[index], 2) + scale * pow(rhs[index], 2));
}

class Tensor
{
public:
    float *data;
    unsigned long shape[NDIM_MAX];
    bool is_cuda;
    int ndims;

private:
    void init(int nargs, ...)
    {
        assert(ndims == nargs - 1); // Already done in constructor
        assert(ndims <= NDIM_MAX);
        va_list vl;
        va_start(vl, nargs);
        float *data_arg = va_arg(vl, float *);
        for (int i = 0; i < ndims; ++i)
            shape[i] = va_arg(vl, int);
        for (int i = ndims; i < NDIM_MAX; ++i)
            shape[i] = 1;
        va_end(vl);
        if (is_cuda) {
            DELETE_CUDA_NULL(data);
            cudaMalloc(&data, numel() * sizeof(float));
            if (data_arg != NULL)
                cudaMemcpy(data, data_arg, numel() * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            DELETE_VEC_NULL(data);
            data = new float[numel()];
            if (data_arg != NULL)
                memcpy(data, data_arg, numel() * sizeof(float));
        }
        assert(data != NULL);
    };

public:
    Tensor() : is_cuda(false), data(NULL), ndims(0){};
    Tensor(int H) : is_cuda(false), data(NULL), ndims(1)
    {
        init(2, (float *)NULL, H);
    }
    Tensor(int H, int W) : is_cuda(false), data(NULL), ndims(2)
    {
        init(3, (float *)NULL, H, W);
    }
    Tensor(int H, int W, int C) : is_cuda(false), data(NULL), ndims(3)
    {
        init(4, (float *)NULL, H, W, C);
    }
    Tensor(int H, int W, int C, int D) : is_cuda(false), data(NULL), ndims(4)
    {
        init(5, (float *)NULL, H, W, C, D);
    }
    // Does copy data from float
    Tensor(float *dat, int H) : is_cuda(false), data(NULL), ndims(1)
    {
        init(2, dat, H);
    }
    Tensor(float *dat, int H, int W) : is_cuda(false), data(NULL), ndims(2)
    {
        init(3, dat, H, W);
    }
    Tensor(float *dat, int H, int W, int C) : is_cuda(false), data(NULL), ndims(3)
    {
        init(4, dat, H, W, C);
    }
    Tensor(float *dat, int H, int W, int C, int D) : is_cuda(false), data(NULL), ndims(4)
    {
        init(5, dat, H, W, C, D);
    }
    ~Tensor()
    {
        if (is_cuda)
            DELETE_CUDA_NULL(data)
        else
            DELETE_VEC_NULL(data)
    }
    Tensor(const Tensor &other)
    {
        for (int i = 0; i < NDIM_MAX; ++i) {
            shape[i] = other.shape[i];
        }
        is_cuda = other.is_cuda;
        ndims = other.ndims;
        if (is_cuda) {
            cudaMalloc(&data, numel() * sizeof(float));
            std::cout << "cuda pointer: " << data << std::endl;
            cudaMemcpy(data, other.data, numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            data = new float[numel()];
            memcpy(data, other.data, numel() * sizeof(float));
        }
    }
    Tensor &operator=(const Tensor &other)
    {
        assert(false);
        return *this;
    }

    bool is_shape(int H, int W, int C, int D)
    {
        return ndims == 4 and shape[0] == H and shape[1] == W and shape[2] == C and shape[3] == D;
    }
    bool is_shape(int H, int W, int C)
    {
        return ndims == 3 and shape[0] == H and shape[1] == W and shape[2] == C;
    }
    bool is_shape(int H, int W)
    {
        return ndims == 2 and shape[0] == H and shape[1] == W;
    }
    bool is_shape(int H)
    {
        return ndims == 1 and shape[0] == H;
    }

    void resize(int H, int W, int C, int D)
    {
        if (!is_shape(H, W, C, D)) {
            ndims = 4;
            init(ndims + 1, (float *)NULL, H, W, C, D);
        }
    }
    void resize(int H, int W, int C)
    {
        if (!is_shape(H, W, C)) {
            ndims = 3;
            init(ndims + 1, (float *)NULL, H, W, C);
        }
    }
    void resize(int H, int W)
    {
        if (!is_shape(H, W)) {
            ndims = 2;
            init(ndims + 1, (float *)NULL, H, W);
        }
    }
    void resize(int H)
    {
        if (!is_shape(H)) {
            ndims = 1;
            init(ndims + 1, (float *)NULL, H);
        }
    }

    unsigned long numel() const
    {
        if (ndims == 0) {
            return 0;
        } else {
            unsigned long res = 1;
            for (int i = 0; i < ndims; ++i)
                res *= shape[i];
            return res;
        }
    }

    void cuda()
    {
        if (!is_cuda) {
            if (numel() > 0) {
                assert(data != NULL);
                float *cuda_data = NULL;
                cudaMalloc((void **)&cuda_data, numel() * sizeof(float));
                cudaMemcpy(cuda_data, data, numel() * sizeof(float), cudaMemcpyHostToDevice);
                delete[] data;
                data = cuda_data;
            } else {
                assert(data == NULL);
            }
            is_cuda = true;
        }
    }

    // ACCESS
    float item() const
    {
        assert(numel() == 1);
        if (is_cuda) {
            float res;
            cudaMemcpy(&res, data, sizeof(float), cudaMemcpyDeviceToHost);
            return res;
        } else
            return data[0];
    }
    float &at(int i, int j, int k, int l)
    {
        assert(is_cuda == false);
        assert(ndims == 4);
        int index = shape[3] * (shape[2] * (shape[1] * i + j) + k) + l;
        return data[index];
    }
    float &at(int i, int j, int k)
    {
        assert(is_cuda == false);
        assert(ndims == 3);
        int index = shape[2] * (shape[1] * i + j) + k;
        return data[index];
    }
    float &at(int i, int j)
    {
        assert(is_cuda == false);
        assert(ndims == 2);
        int index = shape[1] * i + j;
        return data[index];
    }
    float &at(int i)
    {
        assert(is_cuda == false);
        assert(ndims == 1);
        int index = i;
        return data[index];
    }

    // OP
    Tensor &add_(Tensor &other, float scale = 1)
    {
        assert(is_cuda == other.is_cuda);
        assert(numel() == other.numel());
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_add_<<<num_blocks, THREADS_PER_BLOCK>>>(data, other.data, numel(), scale);
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel(); ++i)
                data[i] += scale * other.data[i];
        }
        return *this;
    }
    Tensor &add_(float val)
    {
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_add_<<<num_blocks, THREADS_PER_BLOCK>>>(data, numel(), val);
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel(); ++i)
                data[i] += val;
        }
        return *this;
    }
    Tensor &sub(Tensor &second, Tensor &out, float scale = 1)
    {
        assert(is_cuda == second.is_cuda);
        assert(is_cuda == out.is_cuda);
        assert(numel() == second.numel());
        assert(numel() == out.numel());
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_sub<<<num_blocks, THREADS_PER_BLOCK>>>(out.data, data, second.data, numel(),
                                                          scale);
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel(); ++i)
                out.data[i] = scale * (data[i] - second.data[i]);
        }
        return *this;
    }
    Tensor &copy_(Tensor &other)
    {
        if (is_cuda != other.is_cuda)
            std::cerr << "Warning: Copying across devices" << std::endl;
        assert(numel() == other.numel());
        if (!other.is_cuda)
            copy_(other.data);
        else if (is_cuda)
            cudaMemcpy(data, other.data, numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        else if (!is_cuda)
            cudaMemcpy(data, other.data, numel() * sizeof(float), cudaMemcpyDeviceToHost);
        return *this;
    }
    Tensor &copy_(float *other)
    {
        if (is_cuda)
            cudaMemcpy(data, other, numel() * sizeof(float), cudaMemcpyHostToDevice);
        else
            memcpy(data, other, numel() * sizeof(float));
        return *this;
    }
    Tensor &fill_(float c)
    {
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_fill_<<<num_blocks, THREADS_PER_BLOCK>>>(data, numel(), c);
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel(); ++i) {
                data[i] = c;
            }
        }
        return *this;
    }
    Tensor &mul_(float val)
    {
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_mul_<<<num_blocks, THREADS_PER_BLOCK>>>(data, numel(), val);
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel(); ++i)
                data[i] *= val;
        }
        return *this;
    }
    Tensor &clip_(float start = 0., float end = 255.)
    {
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_clip_<<<num_blocks, THREADS_PER_BLOCK>>>(data, numel(), start, end);
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel(); ++i)
                data[i] = min(max(data[i], start), end);
        }
        return *this;
    }
    Tensor &mag_components_(Tensor &other, float scale = 1)
    {
        assert(is_cuda == other.is_cuda);
        assert(numel() == other.numel());
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_mag_components_<<<num_blocks, THREADS_PER_BLOCK>>>(data, other.data, numel(),
                                                                      scale);
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel(); ++i)
                data[i] = sqrt(pow(data[i], 2) + scale * pow(other.data[i], 2));
        }
        return *this;
    }
    float *get_data_cpu()
    {
        float *cpu_data = NULL;
        if (is_cuda) {
            cpu_data = new float[numel()];
            cudaMemcpy(cpu_data, data, numel() * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            cpu_data = data;
        }
        return cpu_data;
    }

    void minmax(float &min, float &max)
    {
        // TODO: Better Reduction Method
        float *cpu_data = get_data_cpu();
        max = cpu_data[0];
        min = cpu_data[0];
#pragma omp parallel for reduction(max : max), reduction(min : min)
        for (int i = 0; i < numel(); ++i) {
            if ((cpu_data[i] < min))
                min = cpu_data[i];
            if ((cpu_data[i] > max))
                max = cpu_data[i];
        }

        if (is_cuda)
            delete[] cpu_data;
    }
    int arg_minmax(bool max)
    {
        // TODO: Better Reduction Method
        float *cpu_data = get_data_cpu();
        int imax = 0;
        // #pragma omp parallel for
        for (int i = 0; i < numel(); ++i)
            if ((cpu_data[i] < cpu_data[imax]) ^ max)
                imax = i;

        if (is_cuda)
            delete[] cpu_data;

        if (ndims > 1)
            std::cerr << "Warning: Trying to find arg_minmax of high-dimensional Tensor (dim="
                      << ndims << ")" << std::endl;

        return imax;
    }
    int argmax()
    {
        return arg_minmax(true);
    }
    int argmin()
    {
        return arg_minmax(false);
    }
    float sum()
    {
        // TODO: Better Reduction Method
        float *cpu_data = get_data_cpu();
        float ssum = 0;
#pragma omp parallel for reduction(+ : ssum)
        for (int i = 0; i < numel(); ++i)
            ssum += cpu_data[i];
        if (is_cuda)
            delete[] cpu_data;
        return ssum;
    }
    float square_sum()
    {
        // TODO: Better Reduction Method
        float *cpu_data = get_data_cpu();
        float ssum = 0;
#pragma omp parallel for reduction(+ : ssum)
        for (int i = 0; i < numel(); ++i)
            ssum += cpu_data[i] * cpu_data[i];
        if (is_cuda)
            delete[] cpu_data;
        return ssum;
    }
    float exp_sum()
    {
        // TODO: Better Reduction Method
        float *cpu_data = get_data_cpu();
        float ssum = 0;
#pragma omp parallel for reduction(+ : ssum)
        for (int i = 0; i < numel(); ++i)
            ssum += cpu_data[i];
        if (is_cuda)
            delete[] cpu_data;
        return ssum;
    }

    // PRINT
    void print_data(std::ostream &s, float *fdata, int size)
    {
        for (int i = 0; i < size; ++i) {
            s << fdata[i] << " ";
        }
        s << std::endl;
    }
    void print_data(std::ostream &s, float *fdata, int h, int w)
    {
        for (int i = 0; i < h; ++i) {
            print_data(s, fdata + i * w, w);
        }
    }
    void print_data(std::ostream &s, float *fdata, int h, int w, int d)
    {
        int step = w * d;
        for (int i = 0; i < h; ++i) {
            s << "[" << i << ", :, :]" << std::endl;
            print_data(s, fdata + i * step, w, d);
        }
    }
    void print_data(std::ostream &s, float *fdata, int h, int w, int d, int c)
    {
        int step = w * d * c;
        for (int i = 0; i < h; ++i) {
            print_data(s, fdata + i * step, w, d, c);
        }
    }
    void print(std::ostream &s)
    {
        float *cpu_data = get_data_cpu();
        s << "Tensor(";
        for (int i = 0; i < ndims; ++i) {
            s << shape[i];
            if (i != ndims - 1)
                s << ",";
            else
                s << "):" << std::endl;
        }
        switch (ndims) {
            case 1:
                print_data(s, cpu_data, shape[0]);
                break;
            case 2:
                print_data(s, cpu_data, shape[0], shape[1]);
                break;
            case 3:
                print_data(s, cpu_data, shape[0], shape[1], shape[2]);
                break;
            case 4:
                print_data(s, cpu_data, shape[0], shape[1], shape[2], shape[3]);
                break;
            default:
                assert(false);
        }
        if (is_cuda)
            delete[] cpu_data;
    }
};

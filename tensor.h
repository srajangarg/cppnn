#pragma once

#define NDIM_MAX 4

#include <stdarg.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <cassert>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#define DELETE_NULL(x)                                                                             \
    if (x != NULL) {                                                                               \
        delete x;                                                                                  \
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
__global__ void tensor_sub(float *out, float *first, float *second, int size, float scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        out[index] = scale * (first[index] - second[index]);
}
__global__ void tensor_mul_(float *tensor, int size, float c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        tensor[index] *= c;
}

class Tensor
{
public:
    float *data;
    unsigned long shape[NDIM_MAX];
    bool is_cuda;
    int ndims;

private:
    void init(int n, ...)
    {
        assert(ndims == n - 1); // Already done in constructor
        assert(ndims <= NDIM_MAX);
        va_list vl;
        va_start(vl, n);
        data = va_arg(vl, float *);
        for (int i = 0; i < n; ++i)
            shape[i] = va_arg(vl, int);
        for (int i = n; i < NDIM_MAX; ++i)
            shape[i] = 1;
        va_end(vl);
        if (data == NULL)
            data = new float[numel()];
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
    // Does not copy data from float
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
        // TODO: Fix data-sharing problem
        if (data != NULL) {
            if (is_cuda) {
                cudaFree(data);
                data = NULL;
            } else {
                delete[] data;
                data = NULL;
            }
        }
    }

    Tensor &operator=(const Tensor &other)
    {
        assert(false);
        return *this;
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

    unsigned long numel() const
    {
        unsigned long res = 1;
        for (int i = 0; i < ndims; ++i)
            res *= shape[i];
        return res;
    }

    void cuda()
    {
        if (!is_cuda) {
            float *cuda_data = NULL;
            cudaMalloc((void **)&cuda_data, numel() * sizeof(float));
            cudaMemcpy(cuda_data, data, numel() * sizeof(float), cudaMemcpyHostToDevice);

            // std::cout <<  "" __FILE__ ":" << __LINE__ << " freed pointer: " << data << std::endl;
            // delete []data; // Cannot delete because other tensors may share same data
            data = cuda_data;
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
    void add_(Tensor &other, float scale = 1)
    {
        assert(is_cuda == other.is_cuda);
        assert(numel() == other.numel());
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_add_<<<num_blocks, THREADS_PER_BLOCK>>>(data, other.data, numel(), scale);
        } else {
            for (int i = 0; i < numel(); ++i)
                data[i] += scale * other.data[i];
        }
    }
    void sub(Tensor &second, Tensor &out, float scale = 1)
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
            for (int i = 0; i < numel(); ++i)
                out.data[i] = scale * (data[i] - second.data[i]);
        }
    }
    void copy_(Tensor &other)
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
    }
    void copy_(float *other)
    {
        if (is_cuda)
            cudaMemcpy(data, other, numel() * sizeof(float), cudaMemcpyHostToDevice);
        else
            memcpy(data, other, numel() * sizeof(float));
    }
    void fill_(float c)
    {
        if (is_cuda)
            cudaMemset(data, c, numel() * sizeof(float));
        else
            std::fill(data, data + numel(), c);
    }
    void mul_(float c)
    {
        if (is_cuda) {
            int num_blocks = ceil(((float)numel()) / THREADS_PER_BLOCK);
            tensor_mul_<<<num_blocks, THREADS_PER_BLOCK>>>(data, numel(), c);
        } else {
            for (int i = 0; i < numel(); ++i)
                data[i] *= c;
        }
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
    int arg_minmax(bool max)
    {
        // TODO: Better Reduction Method
        float *cpu_data = get_data_cpu();
        int imax = 0;
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
    float sum()
    {
        // TODO: Better Reduction Method
        float *cpu_data = get_data_cpu();
        float ssum = 0;
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
        for (int i = 0; i < numel(); ++i)
            ssum += cpu_data[i];
        if (is_cuda)
            delete[] cpu_data;
        return ssum;
    }
    int argmax()
    {
        return arg_minmax(true);
    }
    int argmin()
    {
        return arg_minmax(false);
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

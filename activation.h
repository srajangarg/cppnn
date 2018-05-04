#pragma once

#include <functional>
#include <cmath>
#include <map>
#include <cuda.h>

inline __host__ __device__ float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

inline __host__ __device__ float sigmoid_d(float x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

inline __host__ __device__ float lrelu(float x)
{
    if (x > 0)
        return x;
    else
        return -0.1 * x;
}

inline __host__ __device__ float lrelu_d(float x)
{
    if (x > 0)
        return 1.0;
    else
        return -0.1;
}

typedef float (*pf)(float a);
enum class Activations { SIGMOID, LRELU };
__device__ pf f_map[2] = {sigmoid, lrelu};
__device__ pf f_d_map[2] = {sigmoid_d, lrelu_d};
pf f_map_host[2] = {sigmoid, lrelu};
pf f_d_map_host[2] = {sigmoid_d, lrelu_d};

__global__ activate(float*inp, float*out, int size, int a) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < size)
        out[index] = scale_vec[index]*f_map[a](inp[index]);
}
__global__ activate_d(float*inp, float*scale_vec, float*out, int size, int a) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < size)
        out[index] = scale_vec[index]*f_d_map[a](inp[index]);
}

class Activation : public Layer
{
public:
    Activation act = 0;

    // TODO get rid of `outs` somehow
    Activation(Activations x, int outs):act(x)
    {
        outputs = outs;
        out_matrix = Tensor(outs);
        dc_dout = Tensor(outs);
    }

    void initialize()
    {
        assert(inputs == outputs);
    }

    void forward()
    {
        if(is_cuda) {
            assert(in_matrix.is_cuda);
            assert(out_matrix.is_cuda);
            assert(out_matrix.numel() == in_matrix.numel());
            int num_blocks = out_matrix.numel()/THREADS_PER_BLOCK;
            activate<<<num_blocks,THREADS_PER_BLOCK>>>(in_matrix, out_matrix, out_matrix.numel(), (int)act);
        }
        else {
            for (int i = 0; i < outputs; i++)
                out_matrix.at(i) = f_map_host[(int)act](in_matrix.at(i));
        }
    }

    void update(float lr)
    {
        // nothing to be done
    }

    void backprop()
    {

        if(is_cuda) {
            assert(in_matrix.is_cuda);
            assert(out_matrix.is_cuda);
            assert(out_matrix.numel() == in_matrix.numel());
            int num_blocks = out_matrix.numel()/THREADS_PER_BLOCK;
            activate_d<<<num_blocks,THREADS_PER_BLOCK>>>(in_matrix, dc_dout, dc_din, out_matrix.numel(), (int)act);
        }
        else {
            for (int i = 0; i < outputs; i++)
                dc_din.at(i) = dc_dout.at(i) * f_d_map_host[(int)act](in_matrix.at(i));
        }
    }
};

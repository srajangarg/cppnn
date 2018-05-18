#pragma once

#include <functional>
#include <cmath>
#include <map>
#include <cassert>

// can be made more efficient by directly placing functions
enum class Errors { MSE, CROSSENTROPY };

inline float mse(Tensor &x, Tensor &tar, Tensor &err, int n)
{
    assert(x.is_cuda == tar.is_cuda);
    assert(x.is_cuda == err.is_cuda);
    if (x.is_cuda) {
        x.sub(tar, err, 2);
        return err.square_sum() / 4;
    } else {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += (x.at(i) - tar.at(i)) * (x.at(i) - tar.at(i));
            err.at(i) = 2 * (x.at(i) - tar.at(i));
        }
        return sum;
    }
}

inline float cross_en(Tensor &x, Tensor &tar, Tensor &err, int n)
{
    assert(x.is_cuda == tar.is_cuda);
    assert(x.is_cuda == err.is_cuda);
    if (x.is_cuda) {
        // Too inefficient for CUDA when n is small
        float *x_cpu_copy = NULL, *tar_cpu_copy = NULL;
        x_cpu_copy = x.get_data_cpu();
        tar_cpu_copy = tar.get_data_cpu();
        float sum = 0;
        for (int i = 0; i < x.numel(); ++i) {
            x_cpu_copy[i] = exp(x_cpu_copy[i]);
            sum += x_cpu_copy[i];
        }
        for (int i = 0; i < x.numel(); ++i)
            x_cpu_copy[i] = x_cpu_copy[i] / sum - tar_cpu_copy[i];

        err.copy_(x_cpu_copy);

        sum = log(sum);
        for (int i = 0; i < n; i++)
            sum -= x_cpu_copy[i] * tar_cpu_copy[i];

        delete[] x_cpu_copy;
        delete[] tar_cpu_copy;
        return sum;
    } else {
        // from http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss
        float sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += exp(x.at(i));

        for (int i = 0; i < n; i++)
            err.at(i) = (exp(x.at(i)) / sum) - tar.at(i);

        sum = log(sum);

        for (int i = 0; i < n; i++)
            sum -= x.at(i) * tar.at(i);
        return sum;
    }
}

std::map<Errors, std::function<float(Tensor &, Tensor &, Tensor &, int)>> e_map
    = {{Errors::MSE, mse}, {Errors::CROSSENTROPY, cross_en}};

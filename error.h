#pragma once

#include <functional>
#include <cmath>
#include <map>
#include <cassert>

// can be made more efficient by directly placing functions
enum class Errors { MSE, CROSSENTROPY };

inline float mse(Tensor& x, Tensor& tar, Tensor& err, int n)
{
    assert(x.is_cuda == tar.is_cuda);
    assert(x.is_cuda == err.is_cuda);
    if (x.is_cuda) {
        // TODO
    }
    else {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += (x.at(i) - tar.at(i)) * (x.at(i) - tar.at(i));
            err.at(i) = 2 * (x.at(i) - tar.at(i));
        }
        return sum;
    }
}

inline float cross_en(Tensor& x, Tensor& tar, Tensor& err, int n)
{

    assert(x.is_cuda == tar.is_cuda);
    assert(x.is_cuda == err.is_cuda);
    if (x.is_cuda) {
        // TODO
    }
    else {
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

std::map<Errors, std::function<float(float *, float *, float *, int)>> e_map
    = {{Errors::MSE, mse}, {Errors::CROSSENTROPY, cross_en}};

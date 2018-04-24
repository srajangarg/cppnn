#pragma once

#include <functional>
#include <cmath>
#include <map>

// can be made more efficient by directly placing functions
enum class Errors { MSE, CROSSENTROPY };

inline float mse(float *x, float *tar, int n)
{
    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += (x[i] - tar[i]) * (x[i] - tar[i]);
    return sum;
}

inline void mse_b(float *x, float *tar, float *err, int n)
{
    for (int i = 0; i < n; i++)
        err[i] = 2 * (x[i] - tar[i]);
}

inline float cross_en(float *x, float *tar, int n)
{
    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += exp(x[i]);
    sum = log(sum);

    for (int i = 0; i < n; i++)
        sum -= x[i] * tar[i];
    return sum;
}

inline void cross_en_b(float *x, float *tar, float *err, int n)
{
    // from http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss
    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += exp(x[i]);

    for (int i = 0; i < n; i++)
        err[i] = (exp(x[i]) / sum) - tar[i];
}

std::map<Errors, std::function<float(float *, float *, int)>> ef_map
    = {{Errors::MSE, mse}, {Errors::CROSSENTROPY, cross_en}};

std::map<Errors, std::function<void(float *, float *, float *, int)>> eb_map
    = {{Errors::MSE, mse_b}, {Errors::CROSSENTROPY, cross_en_b}};

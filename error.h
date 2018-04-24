#pragma once

#include <functional>
#include <cmath>
#include <map>

// can be made more efficient by directly placing functions
enum class Errors { RMSE, MSE, CROSSENTROPY };

inline float rmse(float *x, float *y, int n)
{
    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    sum /= n;
    return sqrt(sum);
}

inline float mse(float *x, float *y, int n)
{
    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    sum /= n;
    return sum;
}

inline float cross_entropy(float *x, float *y, int n)
{
    // from http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss

    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += exp(x[i]);
    sum = log(sum);

    for (int i = 0; i < n; i++)
        sum -= x[i] * y[i];
    return sum;
}

std::map<Errors, std::function<float(float *, float *, int)>> e_map
    = {{Errors::RMSE, rmse}, {Errors::MSE, mse}, {Errors::CROSSENTROPY, cross_entropy}};

#pragma once

#include <functional>
#include <cmath>
#include <map>

enum class Activations { SIGMOID, LRELU };

inline float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

inline float sigmoid_d(float x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

inline float lrelu(float x)
{
    if (x > 0)
        return x;
    else
        return -0.1 * x;
}

inline float lrelu_d(float x)
{
    if (x > 0)
        return 1.0;
    else
        return -0.1;
}

std::map<Activations, std::function<float(float)>> f_map
    = {{Activations::SIGMOID, sigmoid}, {Activations::LRELU, lrelu}};
std::map<Activations, std::function<float(float)>> f_d_map
    = {{Activations::SIGMOID, sigmoid_d}, {Activations::LRELU, lrelu_d}};

class Activation : public Layer
{
public:
    std::function<float(float)> f;
    std::function<float(float)> f_d;

    // TODO get rid of `outs` somehow
    Activation(Activations x, int outs)
    {
        f = f_map[x];
        f_d = f_d_map[x];
        outputs = outs;
        out_matrix = new float[outs];
        dc_dout = new float[outs];
    }

    void initialize()
    {
        assert(inputs == outputs);
    }

    void forward()
    {
        for (int i = 0; i < outputs; i++)
            out_matrix[i] = f(in_matrix[i]);
    }

    void update(float lr)
    {
        // nothing to be done
    }

    void backprop()
    {
        for (int i = 0; i < outputs; i++)
            dc_din[i] = dc_dout[i] * f_d(in_matrix[i]);
    }
};
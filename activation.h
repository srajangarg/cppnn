#pragma once

#include <functional>
#include <cmath>
#include <map>

// can be made more efficient by directly placing functions
enum class Activations { SIGMOID };

inline float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

std::map<Activations, std::function<float(float)>> f_map = {{Activations::SIGMOID, sigmoid}};

class Activation : public Layer
{
public:
    std::function<float(float)> f;

    // TODO get rid of `outs` somehow
    Activation(Activations x, int outs)
    {
        f = f_map[x];
        outputs = outs;
        out_matrix = new float[outs];
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
};
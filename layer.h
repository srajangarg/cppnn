#pragma once

#include <random>
#include "mat.h"

class Layer
{
public:
    virtual void initialize() = 0;
    virtual void forward() = 0;
    virtual void update(float lr) = 0;
    virtual void backprop() = 0;

    // add more common methods and members as and when required

    float *in_matrix;
    float *out_matrix;
    float *dc_dout;
    float *dc_din;
    int outputs;
    int inputs;

    virtual ~Layer()
    {
        // TODO : mostly can be reused!
        delete[] out_matrix;
        delete[] dc_dout;
    }
};

class Dense : public Layer
{

public:
    float *wt_matrix; // input x output
    float *bias;      // 1 x output
    float *dc_dw;     // input x output
    float *dc_dbias;  // 1 x output

    Dense(int outs)
    {
        outputs = outs;
        out_matrix = new float[outs];
        dc_dout = new float[outs];
    }

    void initialize()
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0);

        wt_matrix = new float[inputs * outputs];
        bias = new float[outputs];

        for (int i = 0; i < inputs * outputs; i++)
            wt_matrix[i] = distribution(generator);
        for (int i = 0; i < outputs; i++)
            bias[i] = 0.0;

        dc_dw = new float[inputs * outputs];
        dc_dbias = new float[outputs];
    }

    void forward()
    {
        mat_mul(in_matrix, wt_matrix, out_matrix, 1, inputs, outputs);

        for (int i = 0; i < outputs; i++)
            out_matrix[i] += bias[i];
    }

    void backprop()
    {
        mat_mul(in_matrix, dc_dout, dc_dw, inputs, 1, outputs, true, false);
        mat_mul(dc_dout, wt_matrix, dc_din, 1, outputs, inputs, false, true);
        memcpy(dc_dbias, dc_dout, outputs * sizeof(float));
    }

    void update(float lr)
    {
        for (int i = 0; i < inputs * outputs; i++)
            wt_matrix[i] -= lr * dc_dw[i];
        for (int i = 0; i < outputs; i++)
            bias[i] -= lr * dc_dbias[i];
    }

    ~Dense()
    {
        delete[] wt_matrix;
        delete[] bias;
        delete[] dc_dw;
        delete[] dc_dbias;
    }
};

class Input : public Layer
{
public:
    Input(int outs)
    {
        inputs = -1;
        outputs = outs;
        out_matrix = new float[outs];
        dc_dout = new float[outs];
    }

    void initialize()
    {
        // nothing to be done
    }

    void forward()
    {
        // nothing to be done
    }

    void update(float lr)
    {
        // nothing to be done
    }

    void backprop()
    {
        // nothing to be done
    }
};

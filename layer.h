#pragma once

#include <random>
#include "tensor.h"
#include "mat.h"
#include "mat-cuda.h"

class Layer
{
public:
    virtual void initialize() = 0;
    virtual void forward() = 0;
    virtual void update(float lr) = 0;
    virtual void backprop() = 0;

    // add more common methods and members as and when required

    Tensor in_matrix;
    Tensor out_matrix;
    Tensor dc_dout;
    Tensor dc_din;
    int outputs;
    int inputs;

    virtual ~Layer()
    {
        // TODO : mostly can be reused!
        delete out_matrix;
        delete dc_dout;
    }
};

class Dense : public Layer
{

public:
    Tensor wt_matrix; // input x output
    Tensor bias;      // 1 x output
    Tensor dc_dw;     // input x output
    Tensor dc_dbias;  // 1 x output

    Dense(int outs)
    {
        outputs = outs;
        out_matrix = Tensor(outs);
        dc_dout = Tensor(outs);
    }

    void initialize()
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 0.001);

        wt_matrix = Tensor(inputs, outputs);
        bias = Tensor(outputs);

        for (int i = 0; i < inputs * outputs; i++)
            wt_matrix.data[i] = distribution(generator);
        bias.fill_(0.);

        dc_dw = Tensor(inputs, outputs);
        dc_dbias = Tensor(outputs);
    }

    void forward()
    {
        mat_mul(in_matrix, wt_matrix, out_matrix, 1, inputs, outputs);
        out_matrix.add_(bias);
    }

    void backprop()
    {
        mat_mul(in_matrix, dc_dout, dc_dw, inputs, 1, outputs, true, false);
        mat_mul(dc_dout, wt_matrix, dc_din, 1, outputs, inputs, false, true);
        dc_dbias.copy_(dc_dout);
    }

    void update(float lr)
    {
        wt_matrix.add_(dc_dw, -lr);
        bias.add_(dc_dbias, -lr);
    }

    ~Dense()
    {
        delete wt_matrix;
        delete bias;
        delete dc_dw;
        delete dc_dbias;
    }
};

class Input : public Layer
{
public:
    Input(int outs)
    {
        inputs = -1;
        outputs = outs;
        out_matrix = Tensor(outs);
        dc_dout = Tensor(outs);
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

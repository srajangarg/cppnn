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
    virtual void cuda() = 0;

    // add more common methods and members as and when required
    Tensor *in_matrix = NULL;
    Tensor *out_matrix = NULL;
    Tensor *dc_dout = NULL;
    Tensor *dc_din = NULL;
    int outputs;
    int inputs;
    bool is_cuda = false;

    virtual ~Layer()
    {
        // TODO : mostly can be reused!
        DELETE_NULL(out_matrix);
        DELETE_NULL(dc_dout);
    }
};

class Dense : public Layer
{

public:
    Tensor *wt_matrix = NULL; // input x output
    Tensor *bias = NULL;      // 1 x output
    Tensor *dc_dw = NULL;     // input x output
    Tensor *dc_dbias = NULL;  // 1 x output

    void cuda()
    {
        in_matrix->cuda();
        out_matrix->cuda();
        dc_dout->cuda();
        dc_din->cuda();
        wt_matrix->cuda();
        bias->cuda();
        dc_dw->cuda();
        dc_dbias->cuda();
        is_cuda = true;
    }

    Dense(int outs)
    {
        outputs = outs;
        out_matrix = new Tensor(outs);
        dc_dout = new Tensor(outs);
    }

    void initialize()
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 0.001);

        wt_matrix = new Tensor(inputs, outputs);
        bias = new Tensor(outputs);

        for (int i = 0; i < inputs * outputs; i++)
            wt_matrix->data[i] = distribution(generator);
        bias->fill_(0.);

        dc_dw = new Tensor(inputs, outputs);
        dc_dbias = new Tensor(outputs);
    }

    void forward()
    {
        mat_mul(*in_matrix, *wt_matrix, *out_matrix, 1, inputs, outputs);
        out_matrix->add_(*bias);
    }

    void backprop()
    {
        mat_mul(*in_matrix, *dc_dout, *dc_dw, inputs, 1, outputs, true, false);
        mat_mul(*dc_dout, *wt_matrix, *dc_din, 1, outputs, inputs, false, true);
        dc_dbias->copy_(*dc_dout);
    }

    void update(float lr)
    {
        wt_matrix->add_(*dc_dw, -lr);
        bias->add_(*dc_dbias, -lr);
    }

    ~Dense()
    {
        DELETE_NULL(wt_matrix);
        DELETE_NULL(bias);
        DELETE_NULL(dc_dw);
        DELETE_NULL(dc_dbias);
    }
};

class Input : public Layer
{
public:
    Input(int outs)
    {
        inputs = -1;
        outputs = outs;
        out_matrix = new Tensor(outs);
        dc_dout = new Tensor(outs);
    }

    void cuda()
    {
        // in_matrix->cuda();
        out_matrix->cuda();
        dc_dout->cuda();
        // dc_din->cuda();
        is_cuda = true;
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

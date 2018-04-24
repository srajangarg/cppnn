#pragma once

class Layer
{
public:
    virtual void initialize() = 0;
    virtual void forward() = 0;

    // add more common methods and members as and when required

    float *in_matrix;
    float *out_matrix;
    float *dc_do;
    float *dc_di;
    int outputs;
    int inputs;

    virtual ~Layer()
    {
        delete[] out_matrix;
        delete[] dc_do;
    }
};

class Dense : public Layer
{

public:
    float *wt_matrix; // input x output
    float *bias;      // 1 x output
    float *dc_dwt;    // input x output
    float *dc_dbias;  // 1 x output

    Dense(int outs)
    {
        outputs = outs;
        out_matrix = new float[outs];
        dc_do = new float[outputs];
    }

    void initialize()
    {
        wt_matrix = new float[inputs * outputs];
        bias = new float[outputs];
    }

    void forward()
    {
        mat_mul(in_matrix, wt_matrix, out_matrix, 1, inputs, outputs);

        for (int i = 0; i < outputs; i++)
            out_matrix[i] += bias[i];
    }

    void backprop()
    {
        float* in_t = new float[inputs];

        mat_mul();

        delete in_t;
    }

    ~Dense()
    {
        delete[] wt_matrix;
        delete[] dc_dwt;
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
    }

    void initialize()
    {
        // nothing to be done
    }

    void forward()
    {
        // nothing to be done
    }
};

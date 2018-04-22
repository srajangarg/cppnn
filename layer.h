#pragma once

class Layer
{
public:
    virtual void initialize() = 0;
    virtual void forward() = 0;

    // add more common methods and members as and when required

    float *in_matrix;
    float *out_matrix;
    int outputs;
    int inputs;

    virtual ~Layer()
    {
        delete[] out_matrix;
    }
};

class Dense : public Layer
{

public:
    float *wt_matrix;

    Dense(int outs)
    {
        outputs = outs;
        out_matrix = new float[outs];
    }

    void initialize()
    {
        wt_matrix = new float[inputs * outputs];
        // initialize wt_matrix
    }

    void forward()
    {
        // in_matrix x wt_matrix = out_matrix
    }

    ~Dense()
    {
        delete[] wt_matrix;
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

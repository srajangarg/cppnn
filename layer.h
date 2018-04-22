#pragma once

class Layer
{

    virtual void initialize() = 0;
    virtual void forward() = 0;

    // add more common methods and members as and when required

    float *in_matrix;
    float *out_matrix;
    uint out_size;
    uint in_size;
};

class Dense : public Layer
{

	float * wt_matrix; // in_matrix x wt_matrix = out_matrix

	Dense(uint nodes)
	{
		out_size = nodes;
	}

	void initialize()
	{
		// allocate & initialize wt_matrix
	}


}
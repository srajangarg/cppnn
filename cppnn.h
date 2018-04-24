#pragma once

#include <vector>
#include <assert.h>
#include <cstring>
#include "layer.h"
#include "error.h"

class NN
{
public:
    std::vector<Layer *> layers;
    float *output;

    int inputs, outputs;

    bool training_data_added = false;
    bool validation_data_added = false;
    bool nn_initialized = false;

    float **train_x, **valid_x;
    float **train_y, **valid_y;

    // each data point has `inputs` number of floats
    NN(int ins)
    {
        inputs = ins;
        layers.push_back(new Input(ins));
        output = layers[0]->out_matrix;
        outputs = ins;
    }

    void add_layer(Layer *new_layer)
    {
        auto prev_layer = layers[layers.size() - 1];
        layers.push_back(new_layer);

        new_layer->in_matrix = prev_layer->out_matrix;
        new_layer->inputs = prev_layer->outputs;
        output = new_layer->out_matrix;
        outputs = new_layer->outputs;
    }

    void add_training_data(std::vector<float *> &t_x, std::vector<float *> &t_y)
    {
        // each element of t_x should be a float array having `layers[0]->outputs` elements in it
        // each element of t_y should be a float array having `last_layer->outputs` elements in it
        assert(t_x.size() == t_y.size());

        train_x = t_x.data();
        train_y = t_y.data();

        training_data_added = true;
    }

    void add_validation_data(std::vector<float *> &v_x, std::vector<float *> &v_y)
    {
        // each element of v_x should be a float array having `layers[0]->outputs` elements in it
        // each element of v_y should be a float array having `lasv_layer->outputs` elements in it
        assert(v_x.size() == v_y.size());

        valid_x = v_x.data();
        valid_y = v_y.data();

        validation_data_added = true;
    }

    void initialize()
    {
        for (auto &l : layers)
            l->initialize();
        nn_initialized = true;
    }

    void forward(float *input_data)
    {
        memcpy(layers[0]->out_matrix, input_data, layers[0]->outputs * sizeof(float));

        for (auto &l : layers)
            l->forward();
    }

    void backprop()
    {
    }

    ~NN()
    {
        for (auto &l : layers)
            delete l;
    }
};
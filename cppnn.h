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

    int inputs, outputs;

    bool training_data_added = false;
    bool validation_data_added = false;
    bool nn_initialized = false;

    float **train_x, **train_y;
    int num_train;
    float **valid_x, **valid_y;
    int num_valid;

    float learning_rate;
    std::function<float(float *, float *, int)> errf;
    std::function<void(float *, float *, float *, int)> errb;

    // each data point has `inputs` number of floats
    NN(int ins)
    {
        inputs = ins;
        layers.push_back(new Input(ins));
        outputs = ins;
    }

    void add_layer(Layer *new_layer)
    {
        auto prev_layer = layers[layers.size() - 1];
        layers.push_back(new_layer);

        new_layer->in_matrix = prev_layer->out_matrix;
        new_layer->inputs = prev_layer->outputs;
        new_layer->dc_din = prev_layer->dc_dout;
        outputs = new_layer->outputs;
    }

    void add_training_data(std::vector<float *> &t_x, std::vector<float *> &t_y)
    {
        // each element of t_x should be a float array having `layers[0]->outputs` elements in it
        // each element of t_y should be a float array having `last_layer->outputs` elements in it
        assert(t_x.size() == t_y.size());

        train_x = t_x.data();
        train_y = t_y.data();
        num_train = t_x.size();

        training_data_added = true;
    }

    void add_validation_data(std::vector<float *> &v_x, std::vector<float *> &v_y)
    {
        // each element of v_x should be a float array having `layers[0]->outputs` elements in it
        // each element of v_y should be a float array having `lasv_layer->outputs` elements in it
        assert(v_x.size() == v_y.size());

        valid_x = v_x.data();
        valid_y = v_y.data();
        num_valid = v_x.size();

        validation_data_added = true;
    }

    void initialize(float lr, Errors e)
    {
        for (auto &l : layers)
            l->initialize();

        learning_rate = lr;
        errf = ef_map[e];
        errb = eb_map[e];

        nn_initialized = true;
    }

    void forward(float *input_data)
    {
        auto first_layer = layers[0];
        memcpy(first_layer->out_matrix, input_data, first_layer->outputs * sizeof(float));

        for (auto &l : layers)
            l->forward();
    }

    float backprop(float *target_data)
    {
        auto last_layer = layers[layers.size() - 1];

        errb(last_layer->out_matrix, target_data, last_layer->dc_dout, outputs);
        float err = errf(last_layer->out_matrix, target_data, outputs);

        for (int i = layers.size() - 1; i >= 0; i--)
            layers[i]->backprop();

        return err;
    }

    void train(int epochs)
    {
        if (not nn_initialized) {
            printf("NN not initialized\n");
            return;
        }

        if (not training_data_added) {
            printf("No training data added\n");
            return;
        }

        auto last_layer = layers[layers.size() - 1];

        for (int i = 0; i < epochs; i++) {

            float sum = 0;
            for (int j = 0; j < num_train; j++) {
                forward(train_x[j]);

                // printf("predicted %.4f\n", last_layer->out_matrix[0]);
                // printf("target %.4f\n", train_y[j][0]);

                // printf("error: %.4f\n", backprop(train_y[j]));
                sum += backprop(train_y[j]);

                for (auto &l : layers)
                    l->update(learning_rate);
            }
            printf("error %.4f\n", sum / num_train);
        }
    }

    ~NN()
    {
        for (auto &l : layers)
            delete l;
    }
};
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
    bool is_cuda = false;

    vector<Tensor> train_x, train_y;
    int num_train;
    vector<Tensor> valid_x, valid_y;
    int num_valid;

    float learning_rate;
    std::function<float(Tensor, Tensor, Tensor, int)> e;

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

        int x_size = layers[0]->outputs;
        int y_size = layers.back()->outputs;

        for(auto x:t_x)
            train_x.push_back(Tensor(x,x_size));
        for(auto y:t_y)
            train_y.push_back(Tensor(y,y_size));
        num_train = t_x.size();

        training_data_added = true;
    }

    void add_validation_data(std::vector<float *> &v_x, std::vector<float *> &v_y)
    {
        // each element of v_x should be a float array having `layers[0]->outputs` elements in it
        // each element of v_y should be a float array having `lasv_layer->outputs` elements in it
        assert(v_x.size() == v_y.size());

        int x_size = layers[0]->outputs;
        int y_size = layers.back()->outputs;

        for(auto x:v_x)
            valid_x.push_back(Tensor(x,x_size));
        for(auto y:v_y)
            valid_y.push_back(Tensor(y,y_size));
        num_valid = v_x.size();

        validation_data_added = true;
    }

    void initialize(float lr, Errors err)
    {
        for (auto &l : layers)
            l->initialize();

        learning_rate = lr;
        e = e_map[err];

        nn_initialized = true;
    }

    void forward(Tensor input_data)
    {
        auto first_layer = layers[0];
        first_layer->out_matrix.copy_(input_data)

        for (auto &l : layers)
            l->forward();
    }

    float backprop(Tensor target_data)
    {
        auto last_layer = layers[layers.size() - 1];
        float err = e(last_layer->out_matrix, target_data, last_layer->dc_dout, outputs);

        for (int i = layers.size() - 1; i >= 0; i--)
            layers[i]->backprop();

        return err;
    }

    void validate(void)
    {
        if (!validation_data_added) {
            printf("No validation data set to compute accuracy!\n");
        } else {
            int correct_pred = 0;
            for (int i = 0; i < num_valid; i++) {
                int pred_index, correct_index = 0;
                forward(valid_x[i]);
                for (int j = 0; j < outputs; j++) {
                    if (valid_y[i][j] == 1) {
                        correct_index = j;
                        break;
                    }
                }
                pred_index = (layers[layers.size() - 1]->out_matrix).argmax();
                if (pred_index == correct_index)
                    correct_pred++;
            }
            printf("Validation accuracy = %.4f\n", correct_pred * 1.0 / num_valid);
        }
        return;
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

                // printf("predicted ");
                // for (int i = 0; i < outputs; i++) {
                //     printf("%f, ", last_layer->out_matrix[i]);
                // }
                // printf("\ntarget ");
                // for (int i = 0; i < outputs; i++) {
                //     printf("%f, ", train_y[j][i]);
                // }
                // printf("\n");

                // printf("error: %.4f\n", backprop(train_y[j]));
                sum += backprop(train_y[j]);

                for (auto &l : layers)
                    l->update(learning_rate);
            }
            printf("error %.4f\n", sum / num_train);
            validate();
        }
    }

    ~NN()
    {
        for (auto &l : layers)
            delete l;
    }
};
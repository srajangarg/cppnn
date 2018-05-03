#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "cppnn.h"
#include "activation.h"

using namespace std;

int main(int argc, char const *argv[])
{
    NN nn(784);
    nn.add_layer(new Dense(800));
    nn.add_layer(new Activation(Activations::SIGMOID, 800));
    nn.add_layer(new Dense(10));
    nn.add_layer(new Activation(Activations::SIGMOID, 10));

    nn.initialize(0.0001, Errors::CROSSENTROPY);

    std::ifstream train_file("mnist/mnist_test.csv");
    int train_size = 100;
    // std::ifstream test_file("mnist/mnist_test.csv");
    // int test_Size = 10000;

    std::vector<float *> train_x(train_size), train_y(train_size);

    for (int i = 0; i < train_size; i++) {
        train_x[i] = new float[784];
        train_y[i] = new float[10];
    }

    std::string line;
    int curr_line = 0;
    while (std::getline(train_file, line)) {
        std::stringstream ss(line);
        int i, index = 0;

        for (int j = 0; j < 10; ++j) {
            train_y[curr_line][j] = 0;
        }

        while (ss >> i) {
            if (index == 0) {
                train_y[curr_line][i] = 1;
            } else
                train_x[curr_line][index - 1] = i;

            index++;
            if (ss.peek() == ',')
                ss.ignore();
        }
        curr_line++;
        if (curr_line == train_size) {
            break;
        }
    }

    nn.add_training_data(train_x, train_y);
    nn.train(1000);

    for (int i = 0; i < train_size; i++) {
        delete train_x[i];
        delete train_y[i];
    }

    return 0;
}
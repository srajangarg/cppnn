#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>

#include "cppnn.h"
#include "activation.h"

using namespace std;

int main(int argc, char const *argv[])
{
    struct timeval start_time, end_time;
    float time_in_ms;
    gettimeofday(&start_time, NULL);

    NN nn(784);
    nn.add_layer(new Dense(800));
    nn.add_layer(new Activation(Activations::SIGMOID, 800));
    nn.add_layer(new Dense(10));
    nn.add_layer(new Activation(Activations::SIGMOID, 10));

    nn.initialize(0.01, Errors::CROSSENTROPY);

    std::ifstream train_file("mnist/mnist_test.csv");
    int train_size = 1000;
    // std::ifstream test_file("mnist/mnist_test.csv");
    // int test_Size = 10000;

    std::vector<float *> train_x(train_size), train_y(train_size);

    for (int i = 0; i < train_size; i++) {
        train_x[i] = new float[784];
        train_y[i] = new float[10];
    }

    cout << "Reading data..." << endl;

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

    vector<float *>::const_iterator first = train_x.begin();
    vector<float *>::const_iterator mid = train_x.begin() + (80 * train_size) / 100;
    vector<float *>::const_iterator last = train_x.begin() + train_size;
    vector<float *> train_x_new(first, mid);
    vector<float *> train_x_valid(mid, last);

    first = train_y.begin();
    mid = train_y.begin() + (80 * train_size) / 100;
    last = train_y.begin() + train_size;
    vector<float *> train_y_new(first, mid);
    vector<float *> train_y_valid(mid, last);

    cout << "Adding train/test data..." << endl;

    nn.add_training_data(train_x, train_y);
    nn.add_validation_data(train_x, train_y);

#ifdef CUDA
    cout << "Moving all data to CUDA..." << endl;
    nn.cuda();
#endif

    gettimeofday(&end_time, NULL);
    cout << "Training..." << endl;
    time_in_ms = (end_time.tv_sec - start_time.tv_sec) * 1000
                 + 1.0 * (end_time.tv_usec - start_time.tv_usec) / 1000;

    printf("Time taken in preprocessing = %f ms\n", time_in_ms);
    std::cout << std::endl;

    nn.train(10);

    for (int i = 0; i < train_size; i++) {
        delete train_x[i];
        delete train_y[i];
    }

    return 0;
}
#include "cppnn.h"
#include "activation.h"

int main(int argc, char const *argv[])
{
    NN nn(2);
    nn.add_layer(new Dense(2));
    nn.add_layer(new Activation(Activations::SIGMOID, 2));
    nn.add_layer(new Dense(1));
    nn.add_layer(new Activation(Activations::SIGMOID, 1));

    nn.initialize(1.0, Errors::CROSSENTROPY);

    std::vector<float *> train_x(4), train_y(4);

    for (int i = 0; i < 4; i++) {
        train_x[i] = new float[2];
        train_y[i] = new float;
    }

    train_x[0][0] = 1.0;
    train_x[0][1] = 0.0;
    train_y[0][0] = 1.0;

    train_x[1][0] = 0.0;
    train_x[1][1] = 1.0;
    train_y[1][0] = 0.0;

    train_x[2][0] = -1.0;
    train_x[2][1] = 0.0;
    train_y[2][0] = 1.0;

    train_x[3][0] = 0.0;
    train_x[3][1] = -1.0;
    train_y[3][0] = 0.0;

    nn.add_training_data(train_x, train_y);
    nn.train(200);

    for (int i = 0; i < 4; i++) {
        delete train_x[i];
        delete train_y[i];
    }

    return 0;
}
#pragma once

#include <cassert>

void free_vec(float *v)
{
    if (v != NULL) {
        std::cout << "" __FILE__ ":" __LINE__ " ("__func__") freed pointer: " << v << std::endl;
        delete[] v;
        v = NULL;
    }
}

void alloc_vec(float *&v, int size)
{
    free(v);
    v = new float[size];
    assert(v != NULL);
}

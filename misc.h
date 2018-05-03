#pragma once

#include <cassert>

void free_vec(float *v)
{
    if (v != NULL) {
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

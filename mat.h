#pragma once

#ifndef CUDA

void mat_mul(Tensor &a_t, Tensor &b_t, Tensor &out_t, int m, int k, int n, bool trans_a = false,
             bool trans_b = false)
{
    // op(a): m,k  (op can be transpose or identity)
    // op(b): k,n  (op can be transpose or identity)
    // out: m,n

    assert(!a_t.is_cuda);
    assert(!b_t.is_cuda);
    assert(!out_t.is_cuda);

    float *a = a_t.data, *b = b_t.data, *out = out_t.data;
    assert(a != NULL);
    assert(b != NULL);
    assert(out != NULL);

    int a_index, b_index;
    int a_inc = (trans_a) ? m : 1;
    int b_inc = (trans_b) ? 1 : n;

    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            a_index = (trans_a) ? r : r * k;
            b_index = (trans_b) ? c * k : c;
            out[r * n + c] = 0;
            for (int l = 0; l < k; ++l) {
                out[r * n + c] += a[a_index] * b[b_index];
                a_index += a_inc;
                b_index += b_inc;
            }
        }
    }
}

#endif

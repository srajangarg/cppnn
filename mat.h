#pragma once

void mat_mul(float *a, float *b, float *out, int m, int k, int n, bool trans_a = false,
             bool trans_b = false)
{
    // op(a): m,k  (op can be transpose or identity)
    // op(b): k,n  (op can be transpose or identity)
    // out: m,n
    assert(a != NULL);
    assert(b != NULL);
    assert(out != NULL);

    int a_index, b_index;
    int a_inc = (trans_a) ? m : 1;
    int b_inc = (trans_b) ? 1 : n;

    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            a_index = (trans_a) ? r : r*k;
            b_index = (trans_b) ? c*k : c;
            out[r*n + c] = 0;
            for (int l = 0; l < k; ++l) {
                out[r*n + c] += a[a_index] * b[b_index];
                a_index += a_inc;
                b_index += b_inc;
            }
        }
    }
}

// void mat_trans(float* inp, float* out, int m, int n)
// {
//     // inp: m,n
//     // out: n,m
//     assert(inp != NULL);
//     assert(out != NULL);

//     for (int i = 0; i < m; ++i)
//     {
//     	for (int j = 0; j < n; ++j)
//     	{
//     		out[j*m+i] = inp[i*n+j];
//     	}
//     }
// }

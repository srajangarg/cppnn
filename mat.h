#pragma once

void mat_mul(float* lhs, float* rhs, float* out, int m, int k, int n)
{
    // lhs: m,k
    // rhs: k,n
    // out: m,n
    assert(lhs != NULL);
    assert(rhs != NULL);
    assert(out != NULL);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float res = 0;
            for (int kk = 0; kk < k; ++kk)
            {
                res += lhs[i*k+kk] * rhs[kk*n+j];
            }
            out[i][j] = res;
        }
    }
}

void mat_trans(float* inp, float* out, int m, int n)
{
    // inp: m,n
    // out: n,m
    assert(inp != NULL);
    assert(out != NULL);

    for (int i = 0; i < m; ++i)
    {
    	for (int j = 0; j < n; ++j)
    	{
    		out[j*m+i] = inp[i*n+j];
    	}
    }
}

#pragma once

#ifndef CUDA

inline float &at(float *f, int I, int i)
{
    return f[i];
}

inline float &at(float *f, int I, int J, int i, int j)
{
    return f[J * i + j];
}

inline float &at(float *f, int I, int J, int K, int i, int j, int k)
{
    return f[K * (J * i + j) + k];
}

inline float &at(float *f, int I, int J, int K, int L, int i, int j, int k, int l)
{
    return f[L * (K * (J * i + j) + k) + l];
}

void func_conv2d(float *img, int inF, int H, int W, float *kernel, int outF, int kH, int kW,
                 float *out, bool pad = false)
{
    // img: H, W, inF
    // kernel: outF, kH, kW, inF
    // output: H, W, outF (if pad=true)
    // output: H-kH+1, W-kW+1, outF (if pad=false)

    int kH_centre = kH / 2;
    int kW_centre = kW / 2;

    int outH, outW;
    if (pad) {
        outH = H;
        outW = W;
    } else {
        outH = H - kH + 1;
        outW = W - kW + 1;
    }

    for (int of = 0; of < outF; ++of) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                float res = 0;
                for (int ki = 0; ki < kH; ++ki) {
                    for (int kj = 0; kj < kW; ++kj) {
                        int ii = pad ? (i - kH_centre + ki) : (i + ki);
                        int jj = pad ? (j - kW_centre + kj) : (i + ki);
                        if (ii >= 0 and ii < H and jj >= 0 and jj < W) {
                            for (int ff = 0; ff < inF; ++ff) {
                                res += at(img, H, W, inF, ii, jj, ff)
                                       * at(kernel, outF, kH, kW, inF, of, ki, kj, ff);
                            }
                        }
                    }
                }
                at(out, outH, outW, outF, i, j, of) = res;
            }
        }
    }
}

#endif

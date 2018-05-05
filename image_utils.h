#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include "CImg.hh"
#include "misc.h"

void copy_to_CImg(cil::CImg<float> &cimg, float *img, int H, int W, int C)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < C; ++k) {
                cimg(j, i, 0, k) = at(img, H, W, C, i, j, k);
            }
        }
    }
}

void copy_from_CImg(cil::CImg<float> &cimg, float *img, int H, int W, int C)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < C; ++k) {
                at(img, H, W, C, i, j, k) = cimg(j, i, 0, k);
            }
        }
    }
}

void read_image(const char *fpath, float *&img, int &H, int &W, int &C)
{
    cil::CImg<float> cimg("images/car.jpg");
    H = cimg._height;
    W = cimg._width;
    C = cimg._spectrum;
    img = new float[H * W * C];
    assert(img != NULL);
    copy_from_CImg(cimg, img, H, W, C);
}

void display_image(float *img, int H, int W, int C)
{
    cil::CImg<float> cimg(W, H, 1, C);
    copy_to_CImg(cimg, img, H, W, C);
    cimg.display("image");
}

void save_image(const char *fpath, float *img, int H, int W, int C)
{
    cil::CImg<float> cimg(W, H, 1, C);
    copy_to_CImg(cimg, img, H, W, C);
    cimg.save(fpath);
}

void generate_box_blur_kernel(float *&kernel, int inF, int kH, int kW, int outF)
{
    alloc_vec(kernel, outF * kH * kW * inF);
    for (int of = 0; of < outF; ++of) {
        for (int i = 0; i < kH; ++i) {
            for (int j = 0; j < kW; ++j) {
                for (int ff = 0; ff < inF; ++ff) {
                    at(kernel, outF, kH, kW, inF, of, i, j, ff) = (of == ff) ? 1. / (kH * kW) : 0;
                }
            }
        }
    }
}

void generate_gauss_blur_kernel(float *&kernel, int inF, int kH, int kW, int outF, float sigma = 1)
{
    alloc_vec(kernel, outF * kH * kW * inF);
    float kcH = kH / 2.;
    float kcW = kW / 2.;
    for (int of = 0; of < outF; ++of) {
        float sum = 0;
        for (int i = 0; i < kH; ++i)
            for (int j = 0; j < kW; ++j)
                for (int ff = 0; ff < inF; ++ff)
                    if (of == ff) {
                        float e = exp(-((i - kcH) * (i - kcH) + (j - kcW) * (j - kcW))
                                      / (sigma * sigma));
                        at(kernel, outF, kH, kW, inF, of, i, j, ff) = e;
                        sum += e;
                    } else
                        at(kernel, outF, kH, kW, inF, of, i, j, ff) = 0;

        for (int i = 0; i < kH; ++i)
            for (int j = 0; j < kW; ++j)
                for (int ff = 0; ff < inF; ++ff)
                    at(kernel, outF, kH, kW, inF, of, i, j, ff) /= sum;
    }
}

void generate_highpass_kernel(float *&kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    alloc_vec(kernel, outF * kH * kW * inF);
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            for (int i = 0; i < kH; ++i) {
                for (int j = 0; j < kW; ++j) {
                    if ((i + j) % 2 == 1)
                        at(kernel, outF, kH, kW, inF, of, i, j, ff) = -1;
                    else
                        at(kernel, outF, kH, kW, inF, of, i, j, ff) = 0;
                }
            }
            at(kernel, outF, kH, kW, inF, of, 1, 1, ff) = 4;
        }
    }
}

void generate_sharpen_kernel(float *&kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    alloc_vec(kernel, outF * kH * kW * inF);
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            for (int i = 0; i < kH; ++i) {
                for (int j = 0; j < kW; ++j) {
                    if ((i + j) % 2 == 1)
                        at(kernel, outF, kH, kW, inF, of, i, j, ff) = -1;
                    else
                        at(kernel, outF, kH, kW, inF, of, i, j, ff) = 0;
                }
            }
            at(kernel, outF, kH, kW, inF, of, 1, 1, ff) = 5;
        }
    }
}

void generate_emboss_kernel(float *&kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    alloc_vec(kernel, outF * kH * kW * inF);
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            for (int i = 0; i < kH; ++i) {
                for (int j = 0; j < kW; ++j) {
                    at(kernel, outF, kH, kW, inF, of, i, j, ff) = (i + j) - 2;
                }
            }
            at(kernel, outF, kH, kW, inF, of, 1, 1, ff) = 1;
        }
    }
}

void generate_Gx_kernel(float *&kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    alloc_vec(kernel, outF * kH * kW * inF);
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            at(kernel, outF, kH, kW, inF, of, 0, 0, ff) = -1;
            at(kernel, outF, kH, kW, inF, of, 1, 0, ff) = -2;
            at(kernel, outF, kH, kW, inF, of, 2, 0, ff) = -1;
            at(kernel, outF, kH, kW, inF, of, 0, 1, ff) = 0;
            at(kernel, outF, kH, kW, inF, of, 1, 1, ff) = 0;
            at(kernel, outF, kH, kW, inF, of, 2, 1, ff) = 0;
            at(kernel, outF, kH, kW, inF, of, 0, 2, ff) = 1;
            at(kernel, outF, kH, kW, inF, of, 1, 2, ff) = 2;
            at(kernel, outF, kH, kW, inF, of, 2, 2, ff) = 1;
        }
    }
}

void generate_Gy_kernel(float *&kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    alloc_vec(kernel, outF * kH * kW * inF);
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            at(kernel, outF, kH, kW, inF, of, 0, 0, ff) = -1;
            at(kernel, outF, kH, kW, inF, of, 0, 1, ff) = -2;
            at(kernel, outF, kH, kW, inF, of, 0, 2, ff) = -1;
            at(kernel, outF, kH, kW, inF, of, 1, 0, ff) = 0;
            at(kernel, outF, kH, kW, inF, of, 1, 1, ff) = 0;
            at(kernel, outF, kH, kW, inF, of, 1, 2, ff) = 0;
            at(kernel, outF, kH, kW, inF, of, 2, 0, ff) = 1;
            at(kernel, outF, kH, kW, inF, of, 2, 1, ff) = 2;
            at(kernel, outF, kH, kW, inF, of, 2, 2, ff) = 1;
        }
    }
}

void normalize_image(float *&img, int inF, int H, int W, float end = 255.)
{
    float max_val = -1000000;
    float min_val = 1000000;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int ff = 0; ff < inF; ++ff) {
                max_val = max(max_val, at(img, H, W, inF, i, j, ff));
                min_val = min(min_val, at(img, H, W, inF, i, j, ff));
            }
        }
    }
    std::cout << max_val << " " << min_val << std::endl;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int ff = 0; ff < inF; ++ff) {
                at(img, H, W, inF, i, j, ff)
                    = (at(img, H, W, inF, i, j, ff) - min_val) / max_val * end;
            }
        }
    }
}

void add_image(float *&img1, float *&img2, int inF, int H, int W, float end = 255.)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int ff = 0; ff < inF; ++ff) {
                at(img1, H, W, inF, i, j, ff) += at(img2, H, W, inF, i, j, ff);
            }
        }
    }
}

#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <omp.h>
#include "CImg.hh"
#include "misc.h"
#include "tensor.h"

void copy_to_CImg(cil::CImg<float> &cimg, Tensor &img, int H, int W, int C)
{
    float *img_cpu = img.get_data_cpu();
#pragma omp parallel for collapse(3)
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < C; ++k) {
                cimg(j, i, 0, k) = at(img_cpu, H, W, C, i, j, k);
            }
        }
    }
    if (img.is_cuda) {
        DELETE_VEC_NULL(img_cpu);
    }
}

void copy_from_CImg(cil::CImg<float> &cimg, Tensor &img, int H, int W, int C)
{
    float *img_cpu = img.get_data_cpu();
#pragma omp parallel for collapse(3)
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < C; ++k) {
                at(img_cpu, H, W, C, i, j, k) = cimg(j, i, 0, k);
            }
        }
    }
    if (img.is_cuda) {
        img.copy_(img_cpu);
        DELETE_VEC_NULL(img_cpu);
    }
}

void read_image(const char *fpath, Tensor &img, int &H, int &W, int &C)
{
    cil::CImg<float> cimg(fpath);
    H = cimg._height;
    W = cimg._width;
    C = cimg._spectrum;
    img.resize(H, W, C);
    copy_from_CImg(cimg, img, H, W, C);
}

void display_image(Tensor &img, int H, int W, int C)
{
    cil::CImg<float> cimg(W, H, 1, C);
    copy_to_CImg(cimg, img, H, W, C);
    cimg.display("image");
}

void save_image(const char *fpath, Tensor &img, int H, int W, int C)
{
    cil::CImg<float> cimg(W, H, 1, C);
    copy_to_CImg(cimg, img, H, W, C);
    cimg.save(fpath);
}

void generate_box_blur_kernel(Tensor &kernel, int inF, int kH, int kW, int outF)
{
    kernel.resize(outF, kH, kW, inF);
    float *kernel_cpu = kernel.get_data_cpu();
#pragma omp parallel for collapse(4)
    for (int of = 0; of < outF; ++of) {
        for (int i = 0; i < kH; ++i) {
            for (int j = 0; j < kW; ++j) {
                for (int ff = 0; ff < inF; ++ff) {
                    at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff)
                        = (of == ff) ? 1. / (kH * kW) : 0;
                }
            }
        }
    }
    if (kernel.is_cuda) {
        kernel.copy_(kernel_cpu);
        DELETE_VEC_NULL(kernel_cpu);
    }
}

void generate_gauss_blur_kernel(Tensor &kernel, int inF, int kH, int kW, int outF, float sigma = 1)
{
    kernel.resize(outF, kH, kW, inF);
    float *kernel_cpu = kernel.get_data_cpu();
    float kcH = kH / 2.;
    float kcW = kW / 2.;
    // #pragma omp parallel for
    for (int of = 0; of < outF; ++of) {
        float sum = 0;
#pragma omp parallel for collapse(3), reduction(+ : sum)
        for (int i = 0; i < kH; ++i)
            for (int j = 0; j < kW; ++j)
                for (int ff = 0; ff < inF; ++ff)
                    if (of == ff) {
                        float e = exp(-((i - kcH) * (i - kcH) + (j - kcW) * (j - kcW))
                                      / (sigma * sigma));
                        at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff) = e;
                        sum += e;
                    } else
                        at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff) = 0;

#pragma omp parallel for collapse(3)
        for (int i = 0; i < kH; ++i)
            for (int j = 0; j < kW; ++j)
                for (int ff = 0; ff < inF; ++ff)
                    at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff) /= sum;
    }
    if (kernel.is_cuda) {
        kernel.copy_(kernel_cpu);
        DELETE_VEC_NULL(kernel_cpu);
    }
}

void generate_highpass_kernel(Tensor &kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    assert(inF == outF);
    kernel.resize(outF, kH, kW, inF);
    float *kernel_cpu = kernel.get_data_cpu();
#pragma omp parallel for collapse(4)
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            for (int i = 0; i < kH; ++i) {
                for (int j = 0; j < kW; ++j) {
                    if (ff == of && ((i + j) % 2 == 1))
                        at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff) = -1;
                    else
                        at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff) = 0;
                }
            }
        }
    }
    for (int of = 0; of < outF; ++of)
        at(kernel_cpu, outF, kH, kW, inF, of, 1, 1, of) = 4;

    if (kernel.is_cuda) {
        kernel.copy_(kernel_cpu);
        DELETE_VEC_NULL(kernel_cpu);
    }
}

void generate_sharpen_kernel(Tensor &kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    kernel.resize(outF, kH, kW, inF);
    float *kernel_cpu = kernel.get_data_cpu();
#pragma omp parallel for collapse(4)
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            for (int i = 0; i < kH; ++i) {
                for (int j = 0; j < kW; ++j) {
                    if (ff == of && ((i + j) % 2 == 1))
                        at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff) = -1;
                    else
                        at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff) = 0;
                }
            }
        }
    }
    for (int of = 0; of < outF; ++of)
        at(kernel_cpu, outF, kH, kW, inF, of, 1, 1, of) = 5;

    if (kernel.is_cuda) {
        kernel.copy_(kernel_cpu);
        DELETE_VEC_NULL(kernel_cpu);
    }
}

void generate_emboss_kernel(Tensor &kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    kernel.resize(outF, kH, kW, inF);
    float *kernel_cpu = kernel.get_data_cpu();
#pragma omp parallel for collapse(4)
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            for (int i = 0; i < kH; ++i) {
                for (int j = 0; j < kW; ++j) {
                    at(kernel_cpu, outF, kH, kW, inF, of, i, j, ff) = (i + j) - 2;
                }
            }
        }
    }
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            at(kernel_cpu, outF, kH, kW, inF, of, 1, 1, ff) = 1;
        }
    }
    if (kernel.is_cuda) {
        kernel.copy_(kernel_cpu);
        DELETE_VEC_NULL(kernel_cpu);
    }
}

void generate_Gx_kernel(Tensor &kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    kernel.resize(outF, kH, kW, inF);
    float *kernel_cpu = kernel.get_data_cpu();
#pragma omp parallel for collapse(2)
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            at(kernel_cpu, outF, kH, kW, inF, of, 0, 0, ff) = -1;
            at(kernel_cpu, outF, kH, kW, inF, of, 1, 0, ff) = -2;
            at(kernel_cpu, outF, kH, kW, inF, of, 2, 0, ff) = -1;
            at(kernel_cpu, outF, kH, kW, inF, of, 0, 1, ff) = 0;
            at(kernel_cpu, outF, kH, kW, inF, of, 1, 1, ff) = 0;
            at(kernel_cpu, outF, kH, kW, inF, of, 2, 1, ff) = 0;
            at(kernel_cpu, outF, kH, kW, inF, of, 0, 2, ff) = 1;
            at(kernel_cpu, outF, kH, kW, inF, of, 1, 2, ff) = 2;
            at(kernel_cpu, outF, kH, kW, inF, of, 2, 2, ff) = 1;
        }
    }
    if (kernel.is_cuda) {
        kernel.copy_(kernel_cpu);
        DELETE_VEC_NULL(kernel_cpu);
    }
}

void generate_Gy_kernel(Tensor &kernel, int inF, int kH, int kW, int outF)
{
    assert(kH == 3 and kW == 3);
    kernel.resize(outF, kH, kW, inF);
    float *kernel_cpu = kernel.get_data_cpu();
#pragma omp parallel for collapse(2)
    for (int of = 0; of < outF; ++of) {
        for (int ff = 0; ff < inF; ++ff) {
            at(kernel_cpu, outF, kH, kW, inF, of, 0, 0, ff) = -1;
            at(kernel_cpu, outF, kH, kW, inF, of, 0, 1, ff) = -2;
            at(kernel_cpu, outF, kH, kW, inF, of, 0, 2, ff) = -1;
            at(kernel_cpu, outF, kH, kW, inF, of, 1, 0, ff) = 0;
            at(kernel_cpu, outF, kH, kW, inF, of, 1, 1, ff) = 0;
            at(kernel_cpu, outF, kH, kW, inF, of, 1, 2, ff) = 0;
            at(kernel_cpu, outF, kH, kW, inF, of, 2, 0, ff) = 1;
            at(kernel_cpu, outF, kH, kW, inF, of, 2, 1, ff) = 2;
            at(kernel_cpu, outF, kH, kW, inF, of, 2, 2, ff) = 1;
        }
    }
    if (kernel.is_cuda) {
        kernel.copy_(kernel_cpu);
        DELETE_VEC_NULL(kernel_cpu);
    }
}

inline void normalize_scale_image(Tensor &img, int inF, int H, int W, float end = 255.)
{
    float min_val, max_val;
    img.minmax(min_val, max_val);
    img.add_(-min_val).mul_(end / max_val);
}

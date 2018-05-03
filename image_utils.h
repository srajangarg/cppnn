#pragma once

#include <cassert>
#include "CImg.h"

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
    float *kernel = new float[outF * kH * kW * inF];
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

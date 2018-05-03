#include "conv.h"
#include "conv-cuda.h"
#include "image_utils.h"

int main()
{
    int H, W, inF;
    float *img = NULL;
    read_image("images/car.jpg", img, H, W, inF);

    int kH, kW, outF=3;
    float *kernel = NULL;
    float *img_new = NULL;

    kH=5,kW=5;
    generate_box_blur_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img_new, kH/2, kW/2);
    save_image("box-blur.bmp", img_new, H, W, outF);

    kH=5,kW=5;
    generate_gauss_blur_kernel(kernel, inF, kH, kW, outF, kH);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img_new, kH/2, kW/2);
    save_image("gauss-blur.bmp", img_new, H, W, outF);

    // display_image(img_new, H, W, outF);

    return 0;
}

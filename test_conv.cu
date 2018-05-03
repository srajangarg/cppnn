#include "conv.h"
#include "conv-cuda.h"
#include "image_utils.h"

int main()
{
    int H, W, inF;
    float *img = NULL;
    read_image("images/car.jpg", img, H, W, inF);

    int kH, kW, outF = 3;
    float *kernel = NULL;
    float *img1 = NULL;
    float *img2 = NULL;

    kH = 5, kW = 5;
    generate_box_blur_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    save_image("box-blur.bmp", img1, H, W, outF);

    kH = 5, kW = 5;
    generate_gauss_blur_kernel(kernel, inF, kH, kW, outF, kH);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    save_image("gauss-blur.bmp", img1, H, W, outF);

    kH = 3, kW = 3;
    generate_Gx_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    generate_Gy_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img2, kH / 2, kW / 2);
    add_image(img2, img1, inF, H, W);
    normalize_image(img2, inF, H, W, 255);
    save_image("edge.bmp", img2, H, W, outF);

    kH = 3, kW = 3;
    generate_highpass_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    normalize_image(img1, inF, H, W, 255);
    save_image("highpass.bmp", img1, H, W, outF);

    kH = 3, kW = 3;
    generate_sharpen_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    normalize_image(img1, inF, H, W, 255);
    save_image("sharpen.bmp", img1, H, W, outF);

    kH = 3, kW = 3;
    generate_emboss_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    normalize_image(img1, inF, H, W, 255);
    save_image("emboss.bmp", img1, H, W, outF);

    // display_image(img1, H, W, outF);

    return 0;
}

#include "conv.h"
#include "image_utils.h"

int main()
{
    int H, W, inF;
    Tensor img;
    read_image("images/car.jpg", img, H, W, inF);

    int kH, kW, outF = 3, sigma;
    Tensor kernel;
    Tensor img1(img);
    Tensor img2(img);

#ifdef CUDA
    img.cuda();
    img1.cuda();
    img2.cuda();
    kernel.cuda();
#endif

    kH = 9, kW = 9;
    generate_box_blur_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    save_image("blur-box.bmp", img1, H, W, outF);

    sigma = 5, kH = 13, kW = 13;
    generate_gauss_blur_kernel(kernel, inF, kH, kW, outF, sigma);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    save_image("blur-gauss.bmp", img1, H, W, outF);

    kH = 3, kW = 3;
    generate_Gx_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    generate_Gy_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img2, kH / 2, kW / 2);
    img2.mag_components_(img1);
    normalize_scale_image(img2, inF, H, W, 255);
    save_image("edge.bmp", img2, H, W, outF);

    kH = 3, kW = 3;
    generate_highpass_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    img1.clip_(0, 255);
    save_image("highpass.bmp", img1, H, W, outF);

    kH = 3, kW = 3;
    generate_sharpen_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    img1.clip_(0, 255);
    save_image("sharpen.bmp", img1, H, W, outF);

    kH = 3, kW = 3;
    generate_emboss_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    img1.clip_(0, 255);
    save_image("emboss.bmp", img1, H, W, outF);

    // display_image(img1, H, W, outF);

    return 0;
}

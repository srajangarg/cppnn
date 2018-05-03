#include "conv.h"
#include "conv-cuda.h"
#include "image_utils.h"

int main()
{
    int H, W, inF;
    float *img1_f = NULL;
    read_image("images/car.jpg", img1_f, H, W, inF);

    int kH = 3;
    int kW = 3;
    int outF = 3;
    float *kernel = NULL;
    generate_box_blur_kernel(kernel, inF, kH, kW, outF);

    float *img2_f = new float[H * W * inF];
    func_conv2d(img1_f, inF, H, W, kernel, outF, kH, kW, img2_f);

    // display_image(img2_f, H, W, outF);
    save_image("im2_yay.bmp", img2_f, H, W, outF);

    return 0;
}

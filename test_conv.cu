#include <string>
#include <sys/time.h>
#include "conv.h"
#include "image_utils.h"

float getTimeInMilliSecs(const timeval &t1, const timeval &t2)
{
    return (t2.tv_sec - t1.tv_sec) * 1000
            + 1.0 * (t2.tv_usec - t1.tv_usec) / 1000;
}

int main(int argc, char **argv)
{
    int H, W, inF;
    Tensor img;
    if (argc < 2) {
        printf("Usage: %s  [input-image]\n", argv[0]);
        exit(1);
    }

    timeval start_time, end_time, effect_start_time;
    float time_in_ms;

    std::string filename(argv[1]);
    std::string imagename(filename);

    size_t lastdot = filename.find_last_of(".");
    if (lastdot != std::string::npos)
        imagename = filename.substr(0, lastdot);

    gettimeofday(&start_time, NULL);
    printf("img:%s\n", argv[1]);
    read_image(argv[1], img, H, W, inF);


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
    gettimeofday(&end_time, NULL);

    printf("Preprocessing Time = %f ms\n", getTimeInMilliSecs(start_time, end_time));

    gettimeofday(&effect_start_time, NULL);
    kH = 9, kW = 9;
    generate_box_blur_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    save_image((imagename + "-blur-box.jpg").c_str(), img1, H, W, outF);

    gettimeofday(&end_time, NULL);

    printf("Box Blur Time taken = %f ms\n", getTimeInMilliSecs(effect_start_time, end_time));

    gettimeofday(&effect_start_time, NULL);
    sigma = 5, kH = 13, kW = 13;
    generate_gauss_blur_kernel(kernel, inF, kH, kW, outF, sigma);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    save_image((imagename + "-blur-gauss.jpg").c_str(), img1, H, W, outF);

    gettimeofday(&end_time, NULL);

    printf("Gauss Blur Time taken = %f ms\n", getTimeInMilliSecs(effect_start_time, end_time));

    gettimeofday(&effect_start_time, NULL);
    kH = 3, kW = 3;
    generate_Gx_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    generate_Gy_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img2, kH / 2, kW / 2);
    img2.mag_components_(img1);
    normalize_scale_image(img2, inF, H, W, 255);
    save_image((imagename + "-edge.jpg").c_str(), img2, H, W, outF);

    gettimeofday(&end_time, NULL);

    printf("Edge Detection Time taken = %f ms\n", getTimeInMilliSecs(effect_start_time, end_time));

    gettimeofday(&effect_start_time, NULL);
    kH = 3, kW = 3;
    generate_highpass_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    img1.clip_(0, 255);
    save_image((imagename + "-highpass.jpg").c_str(), img1, H, W, outF);

    gettimeofday(&end_time, NULL);

    printf("High Pass Time taken = %f ms\n", getTimeInMilliSecs(effect_start_time, end_time));

    gettimeofday(&effect_start_time, NULL);
    kH = 3, kW = 3;
    generate_sharpen_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    img1.clip_(0, 255);
    save_image((imagename + "-sharpen.jpg").c_str(), img1, H, W, outF);

    gettimeofday(&end_time, NULL);

    printf("Sharpen Time taken = %f ms\n", getTimeInMilliSecs(effect_start_time, end_time));

    gettimeofday(&effect_start_time, NULL);
    kH = 3, kW = 3;
    generate_emboss_kernel(kernel, inF, kH, kW, outF);
    func_conv2d(img, inF, H, W, kernel, outF, kH, kW, img1, kH / 2, kW / 2);
    img1.clip_(0, 255);
    save_image((imagename + "-emboss.jpg").c_str(), img1, H, W, outF);

    gettimeofday(&end_time, NULL);

    printf("Emboss Time taken = %f ms\n", getTimeInMilliSecs(effect_start_time, end_time));

    gettimeofday(&end_time, NULL);

    printf("Total Time taken = %f ms\n", getTimeInMilliSecs(start_time, end_time));
    // display_image(img1, H, W, outF);

    return 0;
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>


using namespace std;

void createGaussianCUDA(cv::cuda::GpuMat &d_Limg, cv::cuda::GpuMat &d_Rimg, cv::cuda::GpuMat &d_result, int kernel_size, float sigma, int block_x, int block_y);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <method>" << std::endl;
        return -1;
    }

    int kernel_size = std::stoi(argv[1]);
    float sigma = std::stof(argv[2]);

    cv::Mat image = cv::imread("image.jpg");
    cv::Mat leftImage = image(cv::Rect(0, 0, image.cols / 2, image.rows));
    cv::Mat rightImage = image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));

    if (leftImage.empty() || rightImage.empty()) {
        std::cout << "Failed to load input images!" << std::endl;
        return -1;
    }

    cv::Mat BlurImage;

    cv::cuda::GpuMat d_Limg, d_Rimg, d_result;
    
    // for (int block_x = 1; block_x <= 32; block_x *= 2){
        // for (int block_y = 1; block_y <= 32; block_y *= 2){
            auto begin = chrono::high_resolution_clock::now();

            d_Limg.upload(leftImage);
            d_Rimg.upload(rightImage);
            d_result.upload(image);

            createGaussianCUDA(d_Limg, d_Rimg, d_result, kernel_size, sigma, 32, 32);

            d_result.download(BlurImage);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-begin;
            std::cout << "Time: " << diff.count() << " s" << std::endl;
            // std::cout << "Block_x: " << block_x << " Block_y: " << block_y << " Time: " << diff.count() << " s" << std::endl;
        // }
    // }

    cv::imshow("Blurred Image", BlurImage);

    cv::waitKey(0);

    return 0;
}
#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#define CV_PI 3.1415926535897932384626433832795
#define BLOCK_Y 32
#define BLOCK_X 32


__device__ float Gaussian(int x, int y, float sigma){
    return exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * CV_PI * sigma * sigma);
}

__global__ void process(const cv::cuda::PtrStepSz<uchar3> leftImage, const cv::cuda::PtrStepSz<uchar3> rightImage, cv::cuda::PtrStepSz<uchar3> BlurImage, const int kernel_size, const float sigma, int rows, int cols) {
    __shared__ uchar3 leftImage_shared[BLOCK_Y + 50][BLOCK_X + 50];
    __shared__ uchar3 rightImage_shared[BLOCK_Y + 50][BLOCK_X + 50];
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    int i = threadIdx.y + kernel_size/2;
    int j = threadIdx.x + kernel_size/2;

    //inputs into shared memory
    leftImage_shared[i][j] = leftImage(dst_y, dst_x);
    rightImage_shared[i][j] = rightImage(dst_y, dst_x);
    if (threadIdx.y < kernel_size/2){
        leftImage_shared[i-kernel_size/2][j] = leftImage(dst_y-kernel_size/2, dst_x);
        rightImage_shared[i-kernel_size/2][j] = rightImage(dst_y-kernel_size/2, dst_x);
        leftImage_shared[i+blockDim.y][j] = leftImage(dst_y+blockDim.y, dst_x);
        rightImage_shared[i+blockDim.y][j] = rightImage(dst_y+blockDim.y, dst_x);
    }
    if (threadIdx.x < kernel_size/2){
        leftImage_shared[i][j-kernel_size/2] = leftImage(dst_y, dst_x-kernel_size/2);
        rightImage_shared[i][j-kernel_size/2] = rightImage(dst_y, dst_x-kernel_size/2);
        leftImage_shared[i][j+blockDim.x] = leftImage(dst_y, dst_x+blockDim.x);
        rightImage_shared[i][j+blockDim.x] = rightImage(dst_y, dst_x+blockDim.x);
    }
    if (threadIdx.y < kernel_size/2 && threadIdx.x < kernel_size/2){
        leftImage_shared[i-kernel_size/2][j-kernel_size/2] = leftImage(dst_y-kernel_size/2, dst_x-kernel_size/2);
        rightImage_shared[i-kernel_size/2][j-kernel_size/2] = rightImage(dst_y-kernel_size/2, dst_x-kernel_size/2);
        leftImage_shared[i+blockDim.y][j+blockDim.x] = leftImage(dst_y+blockDim.y, dst_x+blockDim.x);
        rightImage_shared[i+blockDim.y][j+blockDim.x] = rightImage(dst_y+blockDim.y, dst_x+blockDim.x);
        leftImage_shared[i+blockDim.y][j-kernel_size/2] = leftImage(dst_y+blockDim.y, dst_x-kernel_size/2);
        rightImage_shared[i+blockDim.y][j-kernel_size/2] = rightImage(dst_y+blockDim.y, dst_x-kernel_size/2);
        leftImage_shared[i-kernel_size/2][j+blockDim.x] = leftImage(dst_y-kernel_size/2, dst_x+blockDim.x);
        rightImage_shared[i-kernel_size/2][j+blockDim.x] = rightImage(dst_y-kernel_size/2, dst_x+blockDim.x);
    }

    __syncthreads();

    if (dst_x < cols && dst_y < rows) {
        float3 Lsum = {0., 0., 0.};
        float3 Rsum = {0., 0., 0.};
        float counter = 0.;
        for (int h=-kernel_size/2;h<=kernel_size/2;h++){
            for (int w=-kernel_size/2;w<=kernel_size/2;w++){
                if (dst_y+h > 0 && dst_y+h < rows && dst_x+w > 0 && dst_x+w < cols && h*h+w*w <= kernel_size*kernel_size){
                    // Lsum += leftImage.at<cv::Vec3b>(i+h, j+w) * Gaussian(h, w, sigma);
                    // Rsum += rightImage.at<cv::Vec3b>(i+h, j+w) * Gaussian(h, w, sigma);
                    Lsum.x += leftImage_shared[i+h][j+w].x * Gaussian(w, h, sigma);
                    Lsum.y += leftImage_shared[i+h][j+w].y * Gaussian(w, h, sigma);
                    Lsum.z += leftImage_shared[i+h][j+w].z * Gaussian(w, h, sigma);
                    Rsum.x += rightImage_shared[i+h][j+w].x * Gaussian(w, h, sigma);
                    Rsum.y += rightImage_shared[i+h][j+w].y * Gaussian(w, h, sigma);
                    Rsum.z += rightImage_shared[i+h][j+w].z * Gaussian(w, h, sigma);
                    counter += Gaussian(w, h, sigma);
                }
            }
        }

        uchar3 BlurLPixel = {(unsigned char)(Lsum.x / counter), (unsigned char)(Lsum.y / counter), (unsigned char)(Lsum.z / counter)};
        BlurImage(dst_y, dst_x) = BlurLPixel;

        uchar3 BlurRPixel = {(unsigned char)(Rsum.x / counter), (unsigned char)(Rsum.y / counter), (unsigned char)(Rsum.z / counter)};
        BlurImage(dst_y, dst_x + cols) = BlurRPixel;
    }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void createGaussianCUDA(cv::cuda::GpuMat& d_Limg, cv::cuda::GpuMat& d_Rimg, cv::cuda::GpuMat& d_result, int kernel_size, float sigma, int block_x, int block_y) {
    const dim3 block(block_x, block_y);
    const dim3 grid(divUp(d_Limg.cols, block.x), divUp(d_Limg.rows, block.y));

    const int rows = d_Limg.rows;
    const int cols = d_Limg.cols;

    process<<<grid, block>>>(d_Limg, d_Rimg, d_result, kernel_size, sigma, rows, cols);
}
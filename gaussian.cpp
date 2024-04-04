#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>


using namespace std;

float Gaussian(int x, int y, float sigma) {
    return exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * CV_PI * sigma * sigma);
}

void GaussianBlur(const cv::Mat& leftImage, const cv::Mat& rightImage, const int kernel_size, const float sigma, cv::Mat& BlurImage) {
    CV_Assert(leftImage.size() == rightImage.size());
    CV_Assert(leftImage.type() == CV_8UC3 && rightImage.type() == CV_8UC3);

    int rows = leftImage.rows;
    int cols = leftImage.cols;

    BlurImage.create(rows, 2*cols, CV_8UC3);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            cv::Vec3f Lsum = {0, 0, 0};
            cv::Vec3f Rsum = {0, 0, 0};
            float counter = 0.;
      		for (int h=-kernel_size/2;h<=kernel_size/2;h++){
      			for (int w=-kernel_size/2;w<=kernel_size/2;w++){
                    if (i+h > 0 && i+h < rows && j+w > 0 && j+w < cols && h*h+w*w <= kernel_size*kernel_size){
                        // Lsum += leftImage.at<cv::Vec3b>(i+h, j+w) * Gaussian(h, w, sigma);
                        // Rsum += rightImage.at<cv::Vec3b>(i+h, j+w) * Gaussian(h, w, sigma);
                        Lsum[2] += leftImage.at<cv::Vec3b>(i+h, j+w)[2] * Gaussian(h, w, sigma);
                        Lsum[1] += leftImage.at<cv::Vec3b>(i+h, j+w)[1] * Gaussian(h, w, sigma);
                        Lsum[0] += leftImage.at<cv::Vec3b>(i+h, j+w)[0] * Gaussian(h, w, sigma);
                        Rsum[2] += rightImage.at<cv::Vec3b>(i+h, j+w)[2] * Gaussian(h, w, sigma);
                        Rsum[1] += rightImage.at<cv::Vec3b>(i+h, j+w)[1] * Gaussian(h, w, sigma);
                        Rsum[0] += rightImage.at<cv::Vec3b>(i+h, j+w)[0] * Gaussian(h, w, sigma);
                        counter += Gaussian(h, w, sigma);
                    }
                }
            }

            cv::Vec3b BlurLPixel;
      		BlurLPixel = Lsum / counter;
            BlurImage.at<cv::Vec3b>(i, j) = BlurLPixel;

            cv::Vec3b BlurRPixel;
            BlurRPixel = Rsum / counter;
            BlurImage.at<cv::Vec3b>(i, j + cols) = BlurRPixel;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <method>" << std::endl;
        return -1;
    }

    // auto begin = chrono::high_resolution_clock::now();

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
    
    omp_set_num_threads(32);
    auto begin = chrono::high_resolution_clock::now(); 

    GaussianBlur(leftImage, rightImage, kernel_size, sigma, BlurImage);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-begin;
    cout << "Time with " << 32 << " threads: " << diff.count() << " s" << endl;

    cv::imshow("Blurred Image", BlurImage);

    cv::waitKey(0);

    return 0;
}
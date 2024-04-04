#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>

using namespace std;

void createAnaglyph(const cv::Mat& leftImage, const cv::Mat& rightImage, const int method, cv::Mat& anaglyphImage) {
    CV_Assert(leftImage.size() == rightImage.size());
    CV_Assert(leftImage.type() == CV_8UC3 && rightImage.type() == CV_8UC3);

    int rows = leftImage.rows;
    int cols = leftImage.cols;

    anaglyphImage.create(rows, cols, CV_8UC3);
    if (method == 1) {
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cv::Vec3b leftPixel = leftImage.at<cv::Vec3b>(i, j);
                cv::Vec3b rightPixel = rightImage.at<cv::Vec3b>(i, j);
                cv::Vec3b anaglyphPixel;
                anaglyphPixel[2] = 0.299*leftPixel[2] + 0.587*leftPixel[1] +0.144*leftPixel[0];
                anaglyphPixel[1] = 0;
                anaglyphPixel[0] = 0.299*rightPixel[2] + 0.587*rightPixel[1] +0.144*rightPixel[0];
                anaglyphImage.at<cv::Vec3b>(i, j) = anaglyphPixel;
            }
        }
    } else if (method == 2) {
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cv::Vec3b leftPixel = leftImage.at<cv::Vec3b>(i, j);
                cv::Vec3b rightPixel = rightImage.at<cv::Vec3b>(i, j);
                cv::Vec3b anaglyphPixel;
                anaglyphPixel[2] = 0.299*leftPixel[2] + 0.587*leftPixel[1] +0.144*leftPixel[0];
                anaglyphPixel[1] = 0.299*rightPixel[2] + 0.587*rightPixel[1] +0.144*rightPixel[0];
                anaglyphPixel[0] = 0.299*rightPixel[2] + 0.587*rightPixel[1] +0.144*rightPixel[0];
                anaglyphImage.at<cv::Vec3b>(i, j) = anaglyphPixel;
            }
        }
    } else if (method == 3) {
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cv::Vec3b leftPixel = leftImage.at<cv::Vec3b>(i, j);
                cv::Vec3b rightPixel = rightImage.at<cv::Vec3b>(i, j);
                cv::Vec3b anaglyphPixel;
                anaglyphPixel[2] = leftPixel[2];
                anaglyphPixel[1] = rightPixel[1];
                anaglyphPixel[0] = rightPixel[0];
                anaglyphImage.at<cv::Vec3b>(i, j) = anaglyphPixel;
            }
        }
    } else if (method == 4) {
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cv::Vec3b leftPixel = leftImage.at<cv::Vec3b>(i, j);
                cv::Vec3b rightPixel = rightImage.at<cv::Vec3b>(i, j);
                cv::Vec3b anaglyphPixel;
                anaglyphPixel[2] = 0.299*leftPixel[2] + 0.587*leftPixel[1] +0.144*leftPixel[0];
                anaglyphPixel[1] = rightPixel[1];
                anaglyphPixel[0] = rightPixel[0];
                anaglyphImage.at<cv::Vec3b>(i, j) = anaglyphPixel;
            }
        }
    } else {
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cv::Vec3b leftPixel = leftImage.at<cv::Vec3b>(i, j);
                cv::Vec3b rightPixel = rightImage.at<cv::Vec3b>(i, j);
                cv::Vec3b anaglyphPixel;
                anaglyphPixel[2] = 0.7*leftPixel[1] +0.3*leftPixel[0];
                anaglyphPixel[1] = rightPixel[1];
                anaglyphPixel[0] = rightPixel[0];
                anaglyphImage.at<cv::Vec3b>(i, j) = anaglyphPixel;
            }
        }
    }
}
    


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <method>" << std::endl;
        return -1;
    }

    int method = std::stoi(argv[1]);

    cv::Mat image = cv::imread("image.jpg");
    cv::Mat leftImage = image(cv::Rect(0, 0, image.cols / 2, image.rows));
    cv::Mat rightImage = image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));

    if (leftImage.empty() || rightImage.empty()) {
        std::cout << "Failed to load input images!" << std::endl;
        return -1;
    }

    cv::Mat anaglyphImage;

    omp_set_num_threads(32);
    auto begin = chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++){
        createAnaglyph(leftImage, rightImage, method, anaglyphImage);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-begin;
    std::cout << "Time: " << diff.count() << " s" << std::endl;
    std::cout << "Time per frame: " << diff.count()/100 << " s" << std::endl;
    std::cout << "IPS: " << 100/diff.count() << std::endl;
    

    cv::imshow("Anaglyph Image", anaglyphImage);

    cv::waitKey();

    return 0;
}
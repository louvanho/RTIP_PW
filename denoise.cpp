#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>
#include <vector>

using namespace std;


double determinantOfCovariance(const cv::Mat& image, int row, int col, int neighborhood) {
    // Define the neighborhood bounds
    int half_neighborhood = neighborhood / 2;
    int start_row = max(0, row - half_neighborhood);
    int end_row = min(image.rows - 1, row + half_neighborhood);
    int start_col = max(0, col - half_neighborhood);
    int end_col = min(image.cols - 1, col + half_neighborhood);

    // Extract the neighborhood pixels
    int sumR = 0, sumG = 0, sumB = 0;
    int counter = 0;
    for (int i = start_row; i <= end_row; ++i) {
        for (int j = start_col; j <= end_col; ++j) {
            sumR += image.at<cv::Vec3b>(i, j)[2];
            sumG += image.at<cv::Vec3b>(i, j)[1];
            sumB += image.at<cv::Vec3b>(i, j)[0];
            counter++;
        }
    }

    // Compute the mean of the neighborhood
    double meanR = static_cast<double>(sumR) / (counter);
    double meanG = static_cast<double>(sumG) / (counter);
    double meanB = static_cast<double>(sumB) / (counter);

    // Compute the covariance matrix
    std::vector<std::vector<double>> covarianceMatrix(3, std::vector<double>(3, 0.0));
    int counter2 = 0;
    for (int i = start_row; i <= end_row; ++i) {
        for (int j = start_col; j <= end_col; ++j) {
            double diffR = image.at<cv::Vec3b>(i, j)[2] - meanR;
            double diffG = image.at<cv::Vec3b>(i, j)[1] - meanG;
            double diffB = image.at<cv::Vec3b>(i, j)[0] - meanB;
            covarianceMatrix[0][0] += diffR * diffR;
            covarianceMatrix[0][1] += diffR * diffG;
            covarianceMatrix[0][2] += diffR * diffB;

            covarianceMatrix[1][1] += diffG * diffR;
            covarianceMatrix[1][2] += diffG * diffG;
            covarianceMatrix[1][2] += diffG * diffB;

            covarianceMatrix[2][0] += diffB * diffR;
            covarianceMatrix[2][1] += diffB * diffG;
            covarianceMatrix[2][2] += diffB * diffB;
            counter2++;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            covarianceMatrix[i][j] /= counter2;
        }
    }

    // Compute the determinant of the covariance matrix
    return covarianceMatrix[0][0] * covarianceMatrix[1][1] * covarianceMatrix[2][2] +
           covarianceMatrix[0][1] * covarianceMatrix[1][2] * covarianceMatrix[2][0] +
           covarianceMatrix[0][2] * covarianceMatrix[1][0] * covarianceMatrix[2][1] -
           covarianceMatrix[0][2] * covarianceMatrix[1][1] * covarianceMatrix[2][0] -
           covarianceMatrix[0][1] * covarianceMatrix[1][0] * covarianceMatrix[2][2] -
           covarianceMatrix[0][0] * covarianceMatrix[1][2] * covarianceMatrix[2][1];
}


void Denoise(const cv::Mat& leftImage, const cv::Mat& rightImage, const int neighborhood, const float ratio, cv::Mat& DenoisedImage) {
    CV_Assert(leftImage.size() == rightImage.size());
    CV_Assert(leftImage.type() == CV_8UC3 && rightImage.type() == CV_8UC3);

    int rows = leftImage.rows;
    int cols = leftImage.cols;

    DenoisedImage.create(rows, 2*cols, CV_8UC3);


    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            float Ldet = determinantOfCovariance(leftImage, i, j, neighborhood);
            float Rdet = determinantOfCovariance(rightImage, i, j, neighborhood);
            Ldet = abs(Ldet);
            Rdet = abs(Rdet);
            int Lkernel_size;
            int Rkernel_size;

            Lkernel_size = ratio / ((Ldet*Ldet/100000)+1);
            Rkernel_size = ratio / ((Rdet*Rdet/100000)+1);

            cv::Vec3f Lsum = {0, 0, 0};
            cv::Vec3f Rsum = {0, 0, 0};
            int Lcounter = 0;
            int Rcounter = 0;
      		for (int h=-Lkernel_size/2;h<=Lkernel_size/2;h++){
      			for (int w=-Lkernel_size/2;w<=Lkernel_size/2;w++){
                    if (i+h > 0 && i+h < rows && j+w > 0 && j+w < cols && h*h+w*w <= Lkernel_size*Lkernel_size){
                        Lsum[2] += leftImage.at<cv::Vec3b>(i+h, j+w)[2];
                        Lsum[1] += leftImage.at<cv::Vec3b>(i+h, j+w)[1];
                        Lsum[0] += leftImage.at<cv::Vec3b>(i+h, j+w)[0];
                        Lcounter ++;
                    }
                }
            }
            for (int h=-Rkernel_size/2;h<=Rkernel_size/2;h++){
                for (int w=-Rkernel_size/2;w<=Rkernel_size/2;w++){
                    if (i+h > 0 && i+h < rows && j+w > 0 && j+w < cols && h*h+w*w <= Rkernel_size*Rkernel_size){
                        Rsum[2] += rightImage.at<cv::Vec3b>(i+h, j+w)[2];
                        Rsum[1] += rightImage.at<cv::Vec3b>(i+h, j+w)[1];
                        Rsum[0] += rightImage.at<cv::Vec3b>(i+h, j+w)[0];
                        Rcounter ++;
                    }
                }
            }

            cv::Vec3b DenoisedLPixel;
      		DenoisedLPixel = Lsum/Lcounter;
            // DenoisedLPixel = {0,0,0};
            DenoisedImage.at<cv::Vec3b>(i, j) = DenoisedLPixel;

            cv::Vec3b DenoisedRPixel;
            DenoisedRPixel = Rsum/Rcounter;
            // DenoisedRPixel = {0,0,0};
            DenoisedImage.at<cv::Vec3b>(i, j + cols) = DenoisedRPixel;

            // DenoisedImage.at<cv::Vec3b>(i, j) = rightImage.at<cv::Vec3b>(i, j);
            // DenoisedImage.at<cv::Vec3b>(i, j + cols) = {Rkernel_size*10, Rkernel_size*10, Rkernel_size*10};
            


        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <method>" << std::endl;
        return -1;
    }

    auto begin = chrono::high_resolution_clock::now();

    int neighborhood = std::stoi(argv[1]);
    float ratio = std::stof(argv[2]);

    cv::Mat image = cv::imread("painting.tif");
    cv::Mat leftImage = image(cv::Rect(0, 0, image.cols / 2, image.rows));
    cv::Mat rightImage = image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));

    if (leftImage.empty() || rightImage.empty()) {
        std::cout << "Failed to load input images!" << std::endl;
        return -1;
    }

    cv::Mat DenoisedImage;
    omp_set_num_threads(48);
    Denoise(leftImage, rightImage, neighborhood, ratio, DenoisedImage);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-begin;

    cv::imshow("Denoised Image", DenoisedImage);

    cout << "Time: " << diff.count() << " s" << endl;

    cv::waitKey(0);

    return 0;
}
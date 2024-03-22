#include <opencv2/opencv.hpp>
#include <iostream>
#include "FisheyeEffect.h"

using namespace cv;
using namespace std;

// Constants for bit depth conversion
const int   BIT_SHIFT = 2;
const float MAX_10BIT = pow(2,10)-1;
const float MAX_8BIT = pow(2, 8) -1;
const float SCALE_FACTOR_10Bit = MAX_10BIT / MAX_8BIT;


// Function to display and save an image
static void displayAndSave10BitImage(const Mat& image_10Bit, const string& windowName) {
    cv::Mat image_8Bit;
    convertScaleAbs(image_10Bit, image_8Bit, 1.0 / 4.0);

    imshow(windowName, image_8Bit);

    // Construct the filename using the window name and ".png" extension
    string filename = windowName + ".png";
    imwrite(filename, image_8Bit);
}

Mat convert10BitTo8Bit(const Mat& image_10Bit) {
    Mat image_8Bit;
    convertScaleAbs(image_10Bit, image_8Bit, 1.0 / 4.0);
    return image_8Bit;
}

Mat convert10BitTo8Bit(const Mat& image_10Bit, bool useScaling) {
    Mat image_8Bit(image_10Bit.size(), CV_8UC1);

    for (int y = 0; y < image_10Bit.rows; ++y) {
        ushort* rowPtr_10bit = (ushort*)image_10Bit.ptr<ushort>(y);
        uchar* rowPtr_8bit = image_8Bit.ptr<uchar>(y);
        for (int x = 0; x < image_10Bit.cols; ++x) {
            rowPtr_8bit[x] = useScaling ? saturate_cast<uchar>(rowPtr_10bit[x] / 4 )                // Use scaling for conversion
                                        : saturate_cast<uchar>(rowPtr_10bit[x] >> BIT_SHIFT);       // Use bit-shifting for conversion
        }
    }
    return image_8Bit;
}


Mat convert8BitTo10Bit(const Mat& image_8Bit, bool useScaling) {
    Mat image_10Bit(image_8Bit.size(), CV_16UC1);
    for (int y = 0; y < image_8Bit.rows; ++y) {
        uchar* rowPtr_8bit = (uchar*)image_8Bit.ptr<uchar>(y);
        ushort* rowPtr_10bit = image_10Bit.ptr<ushort>(y);
        for (int x = 0; x < image_8Bit.cols; ++x) {
            rowPtr_10bit[x] = useScaling ? saturate_cast<ushort>(rowPtr_8bit[x] * SCALE_FACTOR_10Bit) : // Use scaling for conversion
                                           rowPtr_8bit[x] << BIT_SHIFT;                                 // Use bit-shifting for conversion
        }
    }
    return image_10Bit;
}

void calculateErrorMetrics(const Mat& original, const Mat& converted, double& rmse, double& psnr) {
    Mat diff;
    absdiff(original, converted, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    Scalar mse = mean(diff);
    rmse = sqrt(mse[0]);
    psnr = 20 * log10(MAX_10BIT-1) - 10 * log10(mse[0]);
}

void Test10bitTo8BitConversion(cv::Mat img_10bit) {
    // Convert to 8-bit using both methods
    Mat img_8bit_shift = convert10BitTo8Bit(img_10bit, false);
    Mat img_8bit_scale = convert10BitTo8Bit(img_10bit, true);

    // Display and save both images
    displayAndSaveImage(img_8bit_shift, "img_8bit_shift");
    displayAndSaveImage(img_8bit_scale, "img_8bit_scale");

    // Compare the two images

    cout << "10 Bit to 8- Bit Conversion! \n" << endl;

    double maxDiff = cv::norm(img_8bit_shift, img_8bit_scale, NORM_INF);
    if (maxDiff == 0) {
        cout << "The images are exactly the same." << endl;
    }
    else {
        cout << "The images are not the same." << endl;

        Mat img_8bit_diff;
        absdiff(img_8bit_shift, img_8bit_scale, img_8bit_diff);

        cv::threshold(img_8bit_diff, img_8bit_diff, 0, 255, THRESH_BINARY);

        displayAndSaveImage(img_8bit_diff, "img_8bit_diff");
    }
}

int TestConvertscaleAbs() {
    // Create a 10-bit gradient image (1024x100)
    Size ImageSize(MAX_10BIT, 300);
    Mat img_10bit(ImageSize, CV_16UC1);
    for (int y = 0; y < img_10bit.rows; ++y) {
        ushort* rowPtr = img_10bit.ptr<ushort>(y);
        for (int x = 0; x < img_10bit.cols; ++x) {
            rowPtr[x] = x;
        }
    }

    Test10bitTo8BitConversion(img_10bit);

    cout << "8 Bit to 10 - Bit Conversion! \n" << endl;

    Mat img_8bit = convert10BitTo8Bit(img_10bit);
    Mat img_10bit_shift = convert8BitTo10Bit(img_8bit, false);
    Mat img_10bit_scale = convert8BitTo10Bit(img_8bit, true);

    displayAndSave10BitImage(img_10bit_shift, "img_10bit_shift"); 
    displayAndSave10BitImage(img_10bit_scale, "img_10bit_scale");

    double shift_rmse = 0.0;
    double shift_psnr = 0.0;
    double scale_rmse = 0.0;
    double scale_psnr = 0.0;

    calculateErrorMetrics(img_10bit, img_10bit_shift, shift_rmse, shift_psnr);
    calculateErrorMetrics(img_10bit, img_10bit_scale, scale_rmse, scale_psnr);

    Mat img_10bit_diff;
    absdiff(img_10bit_shift, img_10bit_scale, img_10bit_diff);

    cv::threshold(img_10bit_diff, img_10bit_diff, 0, 255, THRESH_BINARY);

    displayAndSaveImage(img_10bit_diff, "img_10bit_diff");

    // Print Results
    cout << "Bit Shifting:\n RMSE: " << shift_rmse << "\n PSNR: " << shift_psnr << endl;
    cout << "Scaling Factor:\n RMSE: " << scale_rmse << "\n PSNR: " << scale_psnr << endl;

    // First, determine the better method based on PSNR and RMSE separately
    string betterMethodPSNR = (shift_psnr > scale_psnr) ? "Bit Shifting" : "Scaling Factor";
    string betterMethodRMSE = (shift_rmse < scale_rmse) ? "Bit Shifting" : "Scaling Factor";

    // Then, calculate the percentage improvement for PSNR and RMSE accurately
    double psnrImprovement = 0.0;
    double rmseImprovement = 0.0;

    if (shift_psnr != scale_psnr) { // Ensure we do not divide by zero
        // For PSNR, higher is better
        psnrImprovement = (betterMethodPSNR == "Bit Shifting") ?
            (shift_psnr - scale_psnr) / scale_psnr * 100 :
            (scale_psnr - shift_psnr) / shift_psnr * 100;
    }

    if (shift_rmse != scale_rmse) { // Ensure we do not divide by zero
        // For RMSE, lower is better
        rmseImprovement = (betterMethodRMSE == "Bit Shifting") ?
            (scale_rmse - shift_rmse) / scale_rmse * 100 :
            (shift_rmse - scale_rmse) / shift_rmse * 100;
    }

    // Print the results
    if (psnrImprovement != 0) {
        cout << "Better method is: " << betterMethodPSNR << endl;
    }

    if (psnrImprovement != 0) {
        cout << "PSNR Improvement: " << abs(psnrImprovement) << "%" << endl;
    }
    else {
        cout << "No PSNR Improvement, both methods are equal." << endl;
    }

    if (rmseImprovement != 0) {
        cout << "RMSE Improvement: " << abs(rmseImprovement) << "%" << " (in favor of the method with better RMSE)" << endl;
    }
    else {
        cout << "No RMSE Improvement, both methods are equal." << endl;
    }

    waitKey(0);
    return 0;
}


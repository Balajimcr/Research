#include "ImageUtils.h"

// ... (Implementations of the clamp function remains the same) ...

// Nearest Neighbor Interpolation
cv::Vec3b ImageUtils::nearestNeighborInterpolation(const cv::Mat& src, double srcX, double srcY) {
    int nearestX = cvRound(srcX);
    int nearestY = cvRound(srcY);

    if (nearestX >= 0 && nearestX < src.cols && nearestY >= 0 && nearestY < src.rows) {
        return src.at<cv::Vec3b>(nearestY, nearestX);
    }
    else {
        // Default behavior for out-of-bounds cases
        return cv::Vec3b(0, 0, 0); // Return black 
    }
}

// Bilinear Interpolation
cv::Vec3b ImageUtils::bilinearInterpolate(const cv::Mat& src, float x, float y) {
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float a = x - x1;
    float b = y - y1;

    cv::Vec3b result(0, 0, 0);
    for (int c = 0; c < 3; ++c) { // Loop over each channel
        if (x1 >= 0 && x2 < src.cols && y1 >= 0 && y2 < src.rows) {
            // Compute bilinear interpolation
            float value = (1 - a) * (1 - b) * src.at<cv::Vec3b>(y1, x1)[c] +
                a * (1 - b) * src.at<cv::Vec3b>(y1, x2)[c] +
                (1 - a) * b * src.at<cv::Vec3b>(y2, x1)[c] +
                a * b * src.at<cv::Vec3b>(y2, x2)[c];
            result[c] = static_cast<uchar>(value);
        }
    }
    return result;
}

// Bicubic Interpolation
cv::Vec3b ImageUtils::bicubicInterpolate(const cv::Mat& src, double srcX, double srcY) {
    int x = static_cast<int>(srcX);
    int y = static_cast<int>(srcY);
    float dx = srcX - x;
    float dy = srcY - y;

    cv::Vec3b result(0, 0, 0);
    for (int channel = 0; channel < 3; ++channel) {
        float sum = 0.0f;
        float weightSum = 0.0f;

        for (int m = -1; m <= 2; ++m) {
            for (int n = -1; n <= 2; ++n) {
                int px = x + n;
                int py = y + m;
                float weight = cubicWeight(n - dx) * cubicWeight(m - dy);

                // Boundary check
                if (px >= 0 && px < src.cols && py >= 0 && py < src.rows) {
                    sum += src.at<cv::Vec3b>(py, px)[channel] * weight;
                    weightSum += weight;
                }
            }
        }

        if (weightSum > 0) {
            result[channel] = static_cast<uchar>(clamp(sum / weightSum, 0.0f, 255.0f));
        }
    }

    return result;
}

// Cubic Weight Calculation
float ImageUtils::cubicWeight(float x) {
    const float a = -0.5; // Cubic coefficient; can be adjusted
    x = std::fabs(x);
    float xSquared = x * x;
    float xCubed = xSquared * x;
    if (x <= 1) {
        return (a + 2) * xCubed - (a + 3) * xSquared + 1;
    }
    else if (x < 2) {
        return a * xCubed - 5 * a * xSquared + 8 * a * x - 4 * a;
    }
    else {
        return 0.0f;
    }
}

// Image Remapping 
void ImageUtils::remap(const cv::Mat& src, cv::Mat& dst,
    const cv::Mat& map1, const cv::Mat& map2,
    InterpolationMethod method) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3); // Now for 3 channel image
    CV_Assert(!map1.empty() && map1.type() == CV_32FC1);
    CV_Assert(!map2.empty() && map2.type() == CV_32FC1);
    CV_Assert(map1.size() == map2.size());

    dst.create(map1.size(), src.type());

    for (int y = 0; y < map1.rows; ++y) {
        for (int x = 0; x < map1.cols; ++x) {
            float srcX = map1.at<float>(y, x);
            float srcY = map2.at<float>(y, x);

            cv::Vec3b interpolatedValue(0, 0, 0);
            switch (method) {
            case InterpolationMethod::NEAREST_NEIGHBOR: // Nearest Neighbor
                interpolatedValue = nearestNeighborInterpolation(src, srcX, srcY);
                break;
            case InterpolationMethod::BILINEAR: // Bilinear
                interpolatedValue = bilinearInterpolate(src, srcX, srcY);
                break;
            case InterpolationMethod::BICUBIC: // Bicubic 
                interpolatedValue = bicubicInterpolate(src, srcX, srcY);
                break;
            default:
                break;
            }

            dst.at<cv::Vec3b>(y, x) = interpolatedValue;
        }
    }
}

#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>

class ImageUtils {
public:
    // Interpolation methods
    enum class InterpolationMethod {
        NEAREST_NEIGHBOR,
        BILINEAR,
        BICUBIC
    };

    // Templated clamping function
    template <typename T>
    static const T& clamp(const T& val, const T& low, const T& high) {
        return std::max(low, std::min(val, high));
    }

    // Interpolation functions
    static cv::Vec3b nearestNeighborInterpolation(const cv::Mat& src, double srcX, double srcY);
    static cv::Vec3b bilinearInterpolate(const cv::Mat& src, float x, float y);
    static cv::Vec3b bicubicInterpolate(const cv::Mat& src, double srcX, double srcY);

    // Cubic weight calculation
    static float cubicWeight(float x);

    // Image remapping utility 
    static void remap(const cv::Mat& src, cv::Mat& dst,
        const cv::Mat& map1, const cv::Mat& map2,
        InterpolationMethod method = InterpolationMethod::BILINEAR);
};

#endif // IMAGE_UTILS_H

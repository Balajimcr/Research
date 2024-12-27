#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>

using namespace cv;
using namespace std;

// Simple clamp for float
inline float clampF(float val, float minVal, float maxVal) {
    return std::max(minVal, std::min(val, maxVal));
}

/**
 * @brief Generate an adaptive grid using Weighted Lloyd's, returning final seeds in integer coords.
 *
 * @param magnitude_of_distortion  [in]  Single-channel float (CV_32FC1) distortion map
 * @param adaptiveGridPoints       [out] Final integer-grid points (cv::Point)
 * @param Grid_x                   [in]  Number of points in x dimension
 * @param Grid_y                   [in]  Number of points in y dimension
 */
void GenerateAdaptiveGrid_WeightedLloyd(
    const Mat& magnitude_of_distortion,
    std::vector<Point>& adaptiveGridPoints,
    int Grid_x,
    int Grid_y
)
{
    CV_Assert(magnitude_of_distortion.type() == CV_32FC1);

    // Total seeds
    const int N = Grid_x * Grid_y;
    CV_Assert(N >= 2);

    const int width = magnitude_of_distortion.cols;
    const int height = magnitude_of_distortion.rows;

    // 1. Create an internal float-based array to store subpixel seeds.
    std::vector<Point2f> seedsFloat;
    seedsFloat.reserve(N);

    // Initialize seeds in a coarse uniform float grid
    const float baseCellWidth = (float)width / (Grid_x - 1);
    const float baseCellHeight = (float)height / (Grid_y - 1);

    for (int i = 0; i < Grid_x; i++) {
        for (int j = 0; j < Grid_y; j++) {
            const float x = clamp(static_cast<int>(i * baseCellWidth), 0, width - 1);
            const float y = clamp(static_cast<int>(j * baseCellHeight), 0, height - 1);
            
            seedsFloat.push_back(Point2f(x, y));
        }
    }

    // Weighted Lloyd’s parameters
    const int   MAX_ITER = 50;
    const float MOVE_THRESHOLD = 1.0f;   // a practical threshold for float
    const float ZERO_WEIGHT_EPS = 1e-12f;

    // Temporary accumulators
    vector<Point2d> wsum(N, Point2d(0, 0));
    vector<double>  wval(N, 0.0);

    // 2. Iterative refinement in float
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        // Reset accumulators
        for (int k = 0; k < N; ++k) {
            wsum[k] = Point2d(0, 0);
            wval[k] = 0.0;
        }

        // 2a. Assign each pixel to nearest seed
        for (int yy = 0; yy < height; ++yy)
        {
            const float* rowPtr = magnitude_of_distortion.ptr<float>(yy);
            for (int xx = 0; xx < width; ++xx)
            {
                float w = rowPtr[xx];
                if (w < 1e-6f) {
                    continue; // skip near-zero
                }

                int nearestIdx = -1;
                float minDist2 = std::numeric_limits<float>::max();

                for (int k = 0; k < N; ++k) {
                    float dx = seedsFloat[k].x - xx;
                    float dy = seedsFloat[k].y - yy;
                    float dist2 = dx * dx + dy * dy;
                    if (dist2 < minDist2) {
                        minDist2 = dist2;
                        nearestIdx = k;
                    }
                }
                wsum[nearestIdx].x += xx * w;
                wsum[nearestIdx].y += yy * w;
                wval[nearestIdx] += w;
            }
        }

        // 2b. Move seeds to weighted centroids
        double maxMovement = 0.0;
        for (int k = 0; k < N; ++k)
        {
            if (wval[k] > ZERO_WEIGHT_EPS) {
                double newX = wsum[k].x / wval[k];
                double newY = wsum[k].y / wval[k];

                double dx = seedsFloat[k].x - newX;
                double dy = seedsFloat[k].y - newY;
                double distMoved = std::sqrt(dx * dx + dy * dy);

                if (distMoved > maxMovement) {
                    maxMovement = distMoved;
                }

                seedsFloat[k].x = (float)newX;
                seedsFloat[k].y = (float)newY;
            }
            else {
                // re-init if dead
                seedsFloat[k].x = float(rand() % width);
                seedsFloat[k].y = float(rand() % height);
            }

            // clamp float coords
            seedsFloat[k].x = clampF(seedsFloat[k].x, 0.f, width - 1.f);
            seedsFloat[k].y = clampF(seedsFloat[k].y, 0.f, height - 1.f);
        }

        // 2c. Convergence check
        if (maxMovement < MOVE_THRESHOLD) {
            break;
        }
    }

    // 3. Convert final seeds (float) --> integer
    //    (round or just cast to int if you prefer truncation)
    adaptiveGridPoints.clear();
    adaptiveGridPoints.reserve(N);
    for (int k = 0; k < N; ++k) {
        int xInt = cvRound(seedsFloat[k].x);  // round or (int) if you want floor
        int yInt = cvRound(seedsFloat[k].y);
        adaptiveGridPoints.push_back(Point(xInt, yInt));
    }
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include "FisheyeEffect.h"
#include "ImageUtils.h"

using namespace cv;
using namespace std;

#define ENABLE_VISUALIZATION


void addPointIfNotPresent(std::vector<cv::Point>& points, const cv::Point& point) {
    if (std::find(points.begin(), points.end(), point) == points.end()) {
        points.push_back(point);
    }
}

// Helper function to find the most frequent segment value in a region
static int findMostFrequentValue(const Mat& segmentedRegion) {
    std::map<int, int> segmentCounts;
    for (int y = 0; y < segmentedRegion.rows; ++y) {
        for (int x = 0; x < segmentedRegion.cols; ++x) {
            segmentCounts[segmentedRegion.at<uchar>(y, x)]++;
        }
    }
    // Find the segment with the maximum count (customize if needed)
    return std::max_element(segmentCounts.begin(), segmentCounts.end(),
        [](const auto& p1, const auto& p2) { return p1.second < p2.second; })->first;
}

static void Generate_FixedGrid(const cv::Mat& distortionMagnitude, std::vector<cv::Point>& fixedGridPoints, int gridX, int gridY) {
    // Constants
    constexpr int DEBUG = 0;

    cv::Mat debugImage;
    if (DEBUG) {
        debugImage = distortionMagnitude.clone();
        if (distortionMagnitude.type() == CV_32FC1) {
            debugImage.convertTo(debugImage, CV_8U, 255);
        }
        cv::cvtColor(debugImage, debugImage, cv::COLOR_GRAY2BGR);
    }

    cv::RNG randomOffset;

    // Calculate cell dimensions
    const float cellWidth = static_cast<float>(distortionMagnitude.cols) / (gridX - 1);
    const float cellHeight = static_cast<float>(distortionMagnitude.rows) / (gridY - 1);

    // Generate grid points
    fixedGridPoints.clear();
    fixedGridPoints.reserve(gridX * gridY);

    for (int i = 0; i < gridX; ++i) {
        for (int j = 0; j < gridY; ++j) {
            int x = static_cast<int>(i * cellWidth);
            int y = static_cast<int>(j * cellHeight);            

            // Boundary checks
            x = clamp(x, 0, distortionMagnitude.cols - 1);
            y = clamp(y, 0, distortionMagnitude.rows - 1);

            // Add to grid points
            fixedGridPoints.emplace_back(x, y);

            if (DEBUG) {
                cv::circle(debugImage, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), 2);
            }
        }
    }

    if (DEBUG) {
        cv::imshow("Fixed Grid Points", debugImage);
    }
}

static void Generate_AdaptiveGrid(const Mat& magnitude_of_distortion, vector<Point>& GDC_Adaptive_Grid_Points, const int Grid_x, const int Grid_y, const float LowThreshold) {
    // Constants for colors
//#define DEBUG_DRAW
#ifdef DEBUG_DRAW
    Mat normalized_magnitude;
    cvtColor(magnitude_of_distortion, normalized_magnitude, COLOR_GRAY2BGR);
    const Scalar Blue(255, 0, 0), Yellow(0, 255, 255), Green(0, 255, 0), Red(0, 0, 255);
#endif

    const int imageWidth = magnitude_of_distortion.cols;
    const int imageHeight = magnitude_of_distortion.rows;

    const float baseCellWidth = static_cast<float>(imageWidth) / static_cast<float>(Grid_x - 1);
    const float baseCellHeight = static_cast<float>(imageHeight) / static_cast<float>(Grid_y - 1);



    // Clear any existing points and reserve space for efficiency
    GDC_Adaptive_Grid_Points.clear();

    for (int i = 0; i < Grid_x; i++) {
        for (int j = 0; j < Grid_y; j++) {
            const int x = clamp(static_cast<int>(i * baseCellWidth), 0, imageWidth - 1);
            const int y = clamp(static_cast<int>(j * baseCellHeight), 0, imageHeight - 1);

            // Add fixed grid points
            GDC_Adaptive_Grid_Points.push_back(Point(x, y));

#ifdef DEBUG_DRAW
            // Debug draw circles for fixed grid points
            circle(normalized_magnitude, Point(x, y), 1, Blue, 2);
#endif
        }
    }

    for (int i = 0; i < Grid_x; ++i) {
        for (int j = 0; j < Grid_y; ++j) {
            const int x = clamp(static_cast<int>(i * baseCellWidth), 0, imageWidth - 1);
            const int y = clamp(static_cast<int>(j * baseCellHeight), 0, imageHeight - 1);

            // Ensure cell boundaries are within image limits
            const float cellWidth = std::min(baseCellWidth, static_cast<float>(imageWidth - x));
            const float cellHeight = std::min(baseCellHeight, static_cast<float>(imageHeight - y));

            const cv::Point CenterPoint(clamp(static_cast<int>(x + (cellWidth / 2.0)), 0, imageWidth - 1),
                clamp(static_cast<int>(y + (cellHeight / 2.0)), 0, imageHeight - 1));
            const cv::Point newPoint(clamp(static_cast<int>(x + (cellWidth / 2.0)), 0, imageWidth - 1), y);
            const cv::Point PointLastRow(clamp(static_cast<int>(x + (cellWidth / 2.0)), 0, imageWidth - 1),
                clamp(static_cast<int>(y + cellHeight), 0, imageHeight - 1));

            // Process distortion values directly
            float predominantValue = magnitude_of_distortion.at<float>(CenterPoint.y, CenterPoint.x);
            if (predominantValue >= LowThreshold) {
                addPointIfNotPresent(GDC_Adaptive_Grid_Points, newPoint);

                if (j == Grid_y - 2) {
                    addPointIfNotPresent(GDC_Adaptive_Grid_Points, PointLastRow);
#ifdef DEBUG_DRAW
                    circle(normalized_magnitude, PointLastRow, 2, Red, 2);
#endif
                }

#ifdef DEBUG_DRAW
                if (predominantValue > 0.9) { // High distortion
                    circle(normalized_magnitude, newPoint, 1, Green, 2);
                }
                else {
                    circle(normalized_magnitude, newPoint, 1, Yellow, 2);
                }
                imshow("4_Adaptive Grid Points", normalized_magnitude);
#endif
            }
        }
    }

#ifdef DEBUG_DRAW
    displayAndSaveImage(normalized_magnitude, "4_Adaptive Grid Points");
    cv::waitKey();
#endif
}
// Helper function to safely get the distortion at floating-point coords
static float getDistortionAt(const Mat& distortionMap, float y, float x)
{
    // Clamp
    int rr = std::max(0, std::min(distortionMap.rows - 1, static_cast<int>(std::floor(y))));
    int cc = std::max(0, std::min(distortionMap.cols - 1, static_cast<int>(std::floor(x))));
    return distortionMap.at<float>(rr, cc);
}

static void GenerateAdaptiveGrid_Random(
    const Mat& magnitude_of_distortion,
    vector<Point>& outPoints,
    const int Grid_x,
    const int Grid_y
)
{
    //-------------------------------------------
    // 1) Basic Setup
    //-------------------------------------------
    const Size ImageSize = magnitude_of_distortion.size();
    // For demonstration, cell size used as baseline minDist
    const Size Cellsize(ImageSize.width / Grid_x, ImageSize.height / Grid_y);

    // Number of points = Grid_x * Grid_y
    const int M = (Grid_x * Grid_y);

    // Distortion thresholds
    // Distortions < lowDistThreshold => "low-distortion area"
    // Distortions >= lowDistThreshold => "high-distortion area"
    const float lowDistThreshold = 0.75f;

    // minDist in low-distortion
    const float minDist = static_cast<float>(Cellsize.width);

    // halfMinDist for points that are BOTH in high-distortion areas
    const float halfMinDist = 0.5f * minDist;

    // Reserve memory
    outPoints.clear();
    outPoints.reserve(M);

    //-------------------------------------------
    // 2) Build prefix sums for sampling
    //-------------------------------------------
    double sumDistortion = 0.0;
    const int rows = magnitude_of_distortion.rows;
    const int cols = magnitude_of_distortion.cols;

    vector<double> prefixSum(rows * cols, 0.0);
    double runningSum = 0.0;
    for (int r = 0; r < rows; ++r) {
        const float* rowPtr = magnitude_of_distortion.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
            runningSum += static_cast<double>(rowPtr[c]);
            prefixSum[r * cols + c] = runningSum;
        }
    }
    sumDistortion = runningSum;

    // If there's effectively no distortion, just place a uniform grid.
    if (sumDistortion <= 1e-12) {
        for (int i = 0; i < M; ++i) {
            float x = static_cast<float>((i % (Grid_x * Grid_y)) / (float)(M - 1)) * (cols - 1);
            float y = static_cast<float>((i / (Grid_x * Grid_y)) / (float)(M - 1)) * (rows - 1);
            outPoints.push_back(Point((int)x, (int)y));
        }
        return;
    }

    //-------------------------------------------
    // 3) Sample M points by "inverse transform sampling"
    //-------------------------------------------
    {
        std::mt19937 rng((unsigned)time(nullptr));
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < M; ++i) {
            double u = dist(rng) * sumDistortion;
            // Binary search in prefixSum
            auto it = std::lower_bound(prefixSum.begin(), prefixSum.end(), u);
            int idx = int(it - prefixSum.begin());
            int rr = idx / cols;
            int cc = idx % cols;

            // Sub-pixel offset
            float x = cc + 0.5f;
            float y = rr + 0.5f;

            // Clamp to image boundary
            if (x >= cols) x = static_cast<float>(cols - 1);
            if (y >= rows) y = static_cast<float>(rows - 1);

            outPoints.push_back(Point(cvRound(x), cvRound(y)));
        }
    }

    //-------------------------------------------
    // 4) (Optional) Weighted Lloyd's Relaxation
    //-------------------------------------------
    {
        const int maxIterations = 10; // tweak as needed
        for (int iter = 0; iter < maxIterations; ++iter) {
            // Create accumulators
            vector<double> sumWeight(M, 0.0);
            vector<double> sumX(M, 0.0);
            vector<double> sumY(M, 0.0);

            // Assign each pixel to the closest center, weighted by distortion
            for (int rr = 0; rr < rows; ++rr) {
                const float* rowPtr = magnitude_of_distortion.ptr<float>(rr);
                for (int cc = 0; cc < cols; ++cc) {
                    float w = rowPtr[cc];
                    if (w <= 0.0f) continue; // skip zero or negative distortions

                    // Find nearest center
                    double minDist2 = 1e30;
                    int bestIdx = 0;
                    for (int k = 0; k < M; ++k) {
                        double dx = outPoints[k].x - cc;
                        double dy = outPoints[k].y - rr;
                        double d2 = dx * dx + dy * dy;
                        if (d2 < minDist2) {
                            minDist2 = d2;
                            bestIdx = k;
                        }
                    }
                    // Accumulate
                    sumWeight[bestIdx] += w;
                    sumX[bestIdx] += w * cc;
                    sumY[bestIdx] += w * rr;
                }
            }

            // Update cluster centers
            for (int k = 0; k < M; ++k) {
                if (sumWeight[k] > 0.0) {
                    float nx = static_cast<float>(sumX[k] / sumWeight[k]);
                    float ny = static_cast<float>(sumY[k] / sumWeight[k]);

                    // clamp
                    nx = std::max(0.0f, std::min(nx, static_cast<float>(cols - 1)));
                    ny = std::max(0.0f, std::min(ny, static_cast<float>(rows - 1)));

                    outPoints[k].x = cvRound(nx);
                    outPoints[k].y = cvRound(ny);
                }
            }
        }
    }

    //-------------------------------------------
    // 5) Enforce a minimum distance in BOTH
    //    low-distortion and high-distortion areas
    //-------------------------------------------
    const int pushIters = 10;  // tweak as needed

    for (int iter = 0; iter < pushIters; ++iter)
    {
        for (int i = 0; i < M; ++i)
        {
            // Distortion at point i
            float di = getDistortionAt(magnitude_of_distortion,
                static_cast<float>(outPoints[i].y),
                static_cast<float>(outPoints[i].x));

            for (int j = i + 1; j < M; ++j)
            {
                // Distortion at point j
                float dj = getDistortionAt(magnitude_of_distortion,
                    static_cast<float>(outPoints[j].y),
                    static_cast<float>(outPoints[j].x));

                // Determine localMinDist based on di, dj
                float localMinDist = 0.0f;
                bool iLow = (di < lowDistThreshold);
                bool jLow = (dj < lowDistThreshold);

                if (iLow && jLow) {
                    // Both in low-distortion => full minDist
                    localMinDist = minDist;
                }
                else if (!iLow && !jLow) {
                    // Both in high-distortion => half minDist
                    localMinDist = halfMinDist;
                }
                else {
                    // One low-dist, one high-dist => pick the bigger distance
                    // so we don’t cause collisions in the low-dist area.
                    localMinDist = minDist;
                }

                float dx = static_cast<float>(outPoints[j].x - outPoints[i].x);
                float dy = static_cast<float>(outPoints[j].y - outPoints[i].y);
                float dist2 = dx * dx + dy * dy;

                // Are they too close?
                float desiredDist2 = localMinDist * localMinDist;
                if (dist2 < desiredDist2 && dist2 > 1e-8f)
                {
                    float distVal = std::sqrt(dist2);
                    float overlap = (localMinDist - distVal) * 0.5f;
                    float ux = dx / distVal;  // x direction
                    float uy = dy / distVal;  // y direction

                    // Push them apart by half the overlap each
                    outPoints[i].x = cvRound(outPoints[i].x - ux * overlap);
                    outPoints[i].y = cvRound(outPoints[i].y - uy * overlap);

                    outPoints[j].x = cvRound(outPoints[j].x + ux * overlap);
                    outPoints[j].y = cvRound(outPoints[j].y + uy * overlap);

                    // Clamp to boundaries
                    outPoints[i].x = std::max(0, std::min(outPoints[i].x, (cols - 1)));
                    outPoints[i].y = std::max(0, std::min(outPoints[i].y, (rows - 1)));
                    outPoints[j].x = std::max(0, std::min(outPoints[j].x, (cols - 1)));
                    outPoints[j].y = std::max(0, std::min(outPoints[j].y, (rows - 1)));
                }
            }
        }
    }
}

void DrawPoints(cv::Mat& image, const std::vector<cv::Point>& points, const cv::Scalar& color, int radius = 2, int thickness = -1) {
    for (const auto& point : points) {
        cv::circle(image, point, radius, color, thickness);
    }
}


void createGridVisualization(const cv::Mat& baseImage, const std::vector<cv::Point>& gridPoints, cv::Mat& outputImage) {
    cv::cvtColor(baseImage, outputImage, cv::COLOR_GRAY2BGR);
    drawGridPoints(gridPoints, outputImage, cv::Scalar(255, 0, 0), 1, 2);
}

void logGridStatistics(int fixedPoints, int adaptivePoints) {
    int pointsDiff = fixedPoints - adaptivePoints;
    double savedPercentage = static_cast<double>(pointsDiff) / fixedPoints * 100;

    std::cout << "Fixed Grid Points: " << fixedPoints
        << ", Adaptive Grid Points: " << adaptivePoints
        << ", Points Saved: " << pointsDiff
        << " (" << savedPercentage << "%)" << std::endl;
}


static void TestAdaptiveGridGeneration() {
    const cv::Size imageSize(1280, 720);
    const int gridX = 30, gridY = 30;
    const int gridX_FG = 33, gridY_FG = 33;
    const float GradientLowThreshold = 0.75;

    // Initialize fisheye distortion
    FisheyeEffect distorter(imageSize);
    distorter.generateDistortionMaps(2.75);

    // Compute distortion magnitude
    cv::Mat mapX, mapY;
    distorter.getDistortionMaps(mapX, mapY);
    const cv::Mat distortionMagnitude = computeDistortionMagnitude(mapX, mapY);

    // Convert to 8-bit for visualization
    cv::Mat distortionMagnitude_8U;
    convertScaleAbs(distortionMagnitude, distortionMagnitude_8U, 255);

    // Generate grid points
    std::vector<cv::Point> fixedGridPoints, adaptiveGridPoints, adaptiveGridPointsQuadTree;
    Generate_FixedGrid(distortionMagnitude, fixedGridPoints, gridX_FG, gridY_FG);
    Generate_AdaptiveGrid(distortionMagnitude, adaptiveGridPoints, gridX, gridY, GradientLowThreshold);
    GenerateAdaptiveGrid_Random(distortionMagnitude, adaptiveGridPointsQuadTree, gridX, gridY);

    // Visualize and draw grids
    cv::Mat fixedGridImage, adaptiveGridImage, adaptiveQuadGridImage;
    createGridVisualization(distortionMagnitude_8U, fixedGridPoints, fixedGridImage);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPoints, adaptiveGridImage);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPointsQuadTree, adaptiveQuadGridImage);

    // Display and save results
    displayAndSaveImage(fixedGridImage, "Fixed Grid Map");
    displayAndSaveImage(adaptiveGridImage, "Adaptive Grid Map");
    displayAndSaveImage(adaptiveQuadGridImage, "Adaptive Quad Grid Map");

    // Calculate and log grid statistics
    logGridStatistics(fixedGridPoints.size(), adaptiveGridPoints.size());

    cv::waitKey();
}


int main() {

    TestAdaptiveGridGeneration();
    return 1;
}
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

#include "FisheyeEffect.h"
#include "ImageUtils.h"

using namespace cv;
using namespace std;

#define ENABLE_VISUALIZATION

// QuadTree Node Structure
struct QuadTreeNode {
    int x0, y0, x1, y1;
    bool isLeaf;
    QuadTreeNode* children[4] = { nullptr, nullptr, nullptr, nullptr };
    int depth;  // Added: store current depth level
};


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
QuadTreeNode* buildQuadTree(const cv::Mat& gradientMap, int x0, int y0, int x1, int y1, float threshold, int currentDepth, int maxDepth) {
    QuadTreeNode* node = new QuadTreeNode{ x0, y0, x1, y1, true, {nullptr, nullptr, nullptr, nullptr}, currentDepth };

    // Stop subdivision if depth limit is reached
    if (currentDepth >= maxDepth) {
        return node;
    }

    // Calculate maximum gradient in the region
    double maxGrad = 0.0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            maxGrad = std::max(maxGrad, static_cast<double>(gradientMap.at<float>(y, x)));
        }
    }

    // Subdivide further if gradient exceeds the threshold
    if (maxGrad > threshold) {
        node->isLeaf = false;

        const int midX = (x0 + x1) / 2;
        const int midY = (y0 + y1) / 2;

        // Recursively build child nodes
        node->children[0] = buildQuadTree(gradientMap, x0, y0, midX, midY, threshold, currentDepth + 1, maxDepth);
        node->children[1] = buildQuadTree(gradientMap, midX, y0, x1, midY, threshold, currentDepth + 1, maxDepth);
        node->children[2] = buildQuadTree(gradientMap, x0, midY, midX, y1, threshold, currentDepth + 1, maxDepth);
        node->children[3] = buildQuadTree(gradientMap, midX, midY, x1, y1, threshold, currentDepth + 1, maxDepth);
    }

    return node;
}

void extractLeafCorners(QuadTreeNode* node, std::vector<cv::Point>& gridPoints) {
    if (!node) return;

    if (node->isLeaf) {

        int width = node->x1 - node->x0;
        int height = node->y1 - node->y0;

        int midX = node->x0 + width / 2;
        int midY = node->y0 + height / 2;

        //gridPoints.emplace_back(midX, midY);
        // Add corners of the leaf region (always)
        gridPoints.emplace_back(node->x0, node->y0);     // Top-left
        //gridPoints.emplace_back(node->x1 - 1, node->y0);     // Top-right
        //gridPoints.emplace_back(node->x0, node->y1 - 1); // Bottom-left
        //gridPoints.emplace_back(node->x1 - 1, node->y1 - 1); // Bottom-right

        // If this leaf is at a depth > 0, it means we subdivided due to high gradient.
        // Add additional points to double the density.
        if (node->depth > 0) {
            // Add mid-edge points
            gridPoints.emplace_back(midX, midY);        // midpoint            
        }
    }
    else {
        for (auto& child : node->children) {
            extractLeafCorners(child, gridPoints);
        }
    }
}

void Generate_UniformAndAdaptiveGrid(const cv::Mat& distortionMagnitude,
    std::vector<cv::Point>& gridPoints,
    int gridX, int gridY, float threshold, int maxDepth) {
    cv::Mat gradientMap = distortionMagnitude.clone();

    const int imageWidth = distortionMagnitude.cols;
    const int imageHeight = distortionMagnitude.rows;

    const float baseCellWidth = static_cast<float>(imageWidth) / (gridX - 1);
    const float baseCellHeight = static_cast<float>(imageHeight) / (gridY - 1);

    // Clear and add a uniform base grid first
    gridPoints.clear();    
    
    // Build QuadTree for adaptive sampling
    QuadTreeNode* root = buildQuadTree(gradientMap, 0, 0, imageWidth, imageHeight, threshold, 0, maxDepth);

    // Extract adaptive points from QuadTree
    std::vector<cv::Point> adaptiveGridPoints;
    extractLeafCorners(root, adaptiveGridPoints);

    // Merge uniform and adaptive points
    for (const auto& pt : adaptiveGridPoints) {
        addPointIfNotPresent(gridPoints, pt);
    }

    // Cleanup memory
    std::function<void(QuadTreeNode*)> deleteQuadTree = [&](QuadTreeNode* node) {
        if (!node) return;
        for (auto& child : node->children) {
            deleteQuadTree(child);
        }
        delete node;
        };
    deleteQuadTree(root);
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
    Generate_UniformAndAdaptiveGrid(distortionMagnitude, adaptiveGridPointsQuadTree, gridX, gridY, GradientLowThreshold, 5);

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
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

// Uncomment to enable debug visualization
//#define DEBUG_DRAW

static inline int clampValue(int v, int low, int high)
{
    return std::max(low, std::min(v, high));
}

static Point2f clampPoint2f(const Point2f& pt, float w, float h)
{
    Point2f cpt(pt.x, pt.y);
    cpt.x = std::max(0.f, std::min(w - 1, cpt.x));
    cpt.y = std::max(0.f, std::min(h - 1, cpt.y));
    return cpt;
}

/**
 * @brief Segment the image by standard nearest-seed approach.
 * @param distortionMap - single-channel CV_32F
 * @param Initial_Grid_Points         - current Initial_Grid_Points
 * @return clusters, where clusters[i] is the list of pixels belonging to Initial_Grid_Points[i].
 */
static vector<vector<Point>> segmentByNearestSeed(
    const Mat& distortionMap,
    const vector<Point2f>& Initial_Grid_Points)
{
    const int rows = distortionMap.rows;
    const int cols = distortionMap.cols;
    vector<vector<Point>> clusters(Initial_Grid_Points.size());

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            float distMin = FLT_MAX;
            int   bestIdx = 0;
            for (size_t s = 0; s < Initial_Grid_Points.size(); ++s)
            {
                float dx = x - Initial_Grid_Points[s].x;
                float dy = y - Initial_Grid_Points[s].y;
                float distSqr = dx * dx + dy * dy;
                if (distSqr < distMin)
                {
                    distMin = distSqr;
                    bestIdx = (int)s;
                }
            }
            clusters[bestIdx].push_back(Point(x, y));
        }
    }
    return clusters;
}

/**
 * @brief Compute the bounding box (min/max) of all pixels in the cluster
 *        whose distortion >= LowThreshold.
 * @return True if we found any pixel above threshold, else false.
 */
static bool computeHighDistortionBoundingBox(
    const Mat& distortionMap,
    const vector<Point>& clusterPixels,
    float LowThreshold,
    int& xMin, int& yMin, int& xMax, int& yMax)
{
    // Initialize bounding box
    xMin = INT_MAX;  xMax = INT_MIN;
    yMin = INT_MAX;  yMax = INT_MIN;

    bool foundAny = false;
    for (const auto& px : clusterPixels)
    {
        float val = distortionMap.at<float>(px.y, px.x);
        if (val >= LowThreshold)
        {
            if (px.x < xMin) xMin = px.x;
            if (px.x > xMax) xMax = px.x;
            if (px.y < yMin) yMin = px.y;
            if (px.y > yMax) yMax = px.y;
            foundAny = true;
        }
    }
    return foundAny;
}

/**
 * @brief Oversample by adding new points at the center of the bounding box
 *        of each high-distortion cluster that exceeds AreaThreshold.
 * @param distortionMap     - input map [0..1]
 * @param Initial_Grid_Points             - (in/out) current Initial_Grid_Points
 * @param targetCount       - desired # of total Initial_Grid_Points
 * @param LowThreshold      - consider pixels above this as "high-distortion"
 * @param AreaThreshold - minimal total cluster distortion to justify a new seed
 * @param maxIterations     - safety limit to avoid infinite loops
 */
static void oversamplePointsInAllHighDistortionClusters(
    const Mat& distortionMap,
    vector<Point2f>& Initial_Grid_Points,
    int    targetCount,
    float  LowThreshold,
    float  AreaThreshold,
    int    maxIterations = 10
)
{
    RNG rng(getTickCount());

    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        if ((int)Initial_Grid_Points.size() >= targetCount)
            break; // already at or above target

        // 1) Segment the image by nearest seed
        vector<vector<Point>> clusters = segmentByNearestSeed(distortionMap, Initial_Grid_Points);

        // 2) Compute weighted area (above LowThreshold) for each cluster
        vector<double> clusterWeightedArea(clusters.size(), 0.0);
        for (size_t c = 0; c < clusters.size(); ++c)
        {
            double areaSum = 0.0;
            for (const auto& px : clusters[c])
            {
                float val = distortionMap.at<float>(px.y, px.x);
                if (val >= LowThreshold)
                    areaSum += val;
            }
            clusterWeightedArea[c] = areaSum;
        }

        // 3) For each cluster above threshold, insert a new point at bounding box center
        bool anyClusterAdded = false;
        for (size_t c = 0; c < clusters.size(); ++c)
        {
            if ((int)Initial_Grid_Points.size() >= targetCount)
                break;

            // If cluster's total distortion is big enough, add a new seed
            if (clusterWeightedArea[c] > AreaThreshold)
            {
                int xMin, yMin, xMax, yMax;
                bool found = computeHighDistortionBoundingBox(
                    distortionMap, clusters[c], LowThreshold, xMin, yMin, xMax, yMax
                );

                if (found)
                {
                    float centerX = 0.5f * (xMin + xMax);
                    float centerY = 0.5f * (yMin + yMax);

                    // Optional: small random jitter to avoid duplicates
                    float jitterX = rng.uniform(-0.5f, 0.5f);
                    float jitterY = rng.uniform(-0.5f, 0.5f);

                    Point2f newSeed(centerX + jitterX, centerY + jitterY);

                    // clamp to valid image coords
                    newSeed = clampPoint2f(newSeed, (float)distortionMap.cols, (float)distortionMap.rows);

                    Initial_Grid_Points.push_back(newSeed);
                    anyClusterAdded = true;
                }
            }
        }

        // If in this iteration we didn't add any cluster, no point continuing
        if (!anyClusterAdded)
            break;
    }
}

/**
 * @brief Weighted Voronoi / Lloyd-like relaxation (for reference).
 *        This moves existing Initial_Grid_Points to better positions, but does NOT add new Initial_Grid_Points.
 */
static void approximateWeightedVoronoi(
    const Mat& distortionMap,
    vector<Point2f>& Initial_Grid_Points,
    int   iterations,
    float alpha,
    float LowThreshold)
{
    const int width = distortionMap.cols;
    const int height = distortionMap.rows;

    for (int iter = 0; iter < iterations; ++iter)
    {
        // Create clusters for each seed
        vector<vector<Point>> clusters(Initial_Grid_Points.size());

        // 1) Assign each pixel to the closest seed (weighted distance)
        for (int y = 0; y < height; ++y)
        {
            const float* dptr = distortionMap.ptr<float>(y);
            for (int x = 0; x < width; ++x)
            {
                float weightVal = (dptr[x] < LowThreshold) ? 0.f : dptr[x];
                float distMin = FLT_MAX;
                int   bestIdx = 0;
                for (size_t s = 0; s < Initial_Grid_Points.size(); ++s)
                {
                    float dx = x - Initial_Grid_Points[s].x;
                    float dy = y - Initial_Grid_Points[s].y;
                    float distSqr = dx * dx + dy * dy;
                    float wdist = distSqr / (1.f + alpha * weightVal);
                    if (wdist < distMin)
                    {
                        distMin = wdist;
                        bestIdx = (int)s;
                    }
                }
                clusters[bestIdx].push_back(Point(x, y));
            }
        }

        // 2) Move each seed to the weighted centroid of its cluster
        for (size_t s = 0; s < Initial_Grid_Points.size(); ++s)
        {
            if (!clusters[s].empty())
            {
                double sumX = 0.0, sumY = 0.0;
                double wSum = 0.0;
                for (const auto& px : clusters[s])
                {
                    float val = distortionMap.at<float>(px.y, px.x);
                    if (val < LowThreshold) val = 0.f;
                    sumX += px.x * val;
                    sumY += px.y * val;
                    wSum += val;
                }
                if (wSum > 1e-12)
                {
                    Initial_Grid_Points[s].x = (float)(sumX / wSum);
                    Initial_Grid_Points[s].y = (float)(sumY / wSum);
                }
                // clamp
                Initial_Grid_Points[s] = clampPoint2f(Initial_Grid_Points[s], (float)width, (float)height);
            }
        }
    }
}

void GenerateAdaptiveGrid_HAG_WeightedVoronoi(
    const Mat& magnitude_of_distortion,
    vector<Point>& GDC_Adaptive_Grid_Points,
    const int Grid_x,
    const int Grid_y,
    const float LowThreshold
)
{
    // We want exactly Grid_x * Grid_y points (the "target" if we do not exceed it)
    const int targetCount = Grid_x * Grid_y;
    const cv::Point Grid_Init(Grid_x * 0.90, Grid_y * 0.90);

    const int width = magnitude_of_distortion.cols;
    const int height = magnitude_of_distortion.rows;

#ifdef DEBUG_DRAW
    Mat debugImage;
    {
        // Convert to BGR for visualization
        Mat tmp;
        magnitude_of_distortion.convertTo(tmp, CV_8UC1, 255.0);
        cvtColor(tmp, debugImage, COLOR_GRAY2BGR);
    }
#endif

    //--------------------------------------------------------------------------
    // 1) Initialize Initial_Grid_Points uniformly (basic grid)
    //--------------------------------------------------------------------------
    vector<Point2f> Initial_Grid_Points;
    Initial_Grid_Points.reserve(targetCount);
    for (int i = 0; i < Grid_Init.x; i++)
    {
        for (int j = 0; j < Grid_Init.y; j++)
        {
            float x = (float)i * (width - 1) / (float)(Grid_x - 1);
            float y = (float)j * (height - 1) / (float)(Grid_y - 1);
            Initial_Grid_Points.emplace_back(x, y);
        }
    }

    //--------------------------------------------------------------------------
    // 2) (Optional) Weighted Voronoi Relaxation 
    //    - This moves existing Initial_Grid_Points to better positions.
    //--------------------------------------------------------------------------
    approximateWeightedVoronoi(
        magnitude_of_distortion,
        Initial_Grid_Points,
        /*iterations=*/10,
        /*alpha=*/20.f,
        LowThreshold
    );

    //--------------------------------------------------------------------------
    // 3) Oversample: Insert *new* Initial_Grid_Points at the center of bounding boxes in 
    //    all high-distortion clusters exceeding some threshold.
    //--------------------------------------------------------------------------
    float AreaThreshold = 0.50;
    // Adjust this based on image size and typical distortion values 
    // so that only truly "large" high-distortion clusters get new Initial_Grid_Points.

    oversamplePointsInAllHighDistortionClusters(
        magnitude_of_distortion,
        Initial_Grid_Points,
        targetCount,
        LowThreshold,
        AreaThreshold,
        /*maxIterations=*/10
    );

    //--------------------------------------------------------------------------
    // 4) If we exceed the targetCount, you might do an undersample pass
    //    or keep the extra points. For brevity, not shown here.
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // 5) Convert float Initial_Grid_Points to integer points
    //--------------------------------------------------------------------------
    GDC_Adaptive_Grid_Points.clear();
    GDC_Adaptive_Grid_Points.reserve(Initial_Grid_Points.size());
    for (const auto& s : Initial_Grid_Points)
    {
        int ix = clampValue((int)std::round(s.x), 0, width - 1);
        int iy = clampValue((int)std::round(s.y), 0, height - 1);
        GDC_Adaptive_Grid_Points.emplace_back(ix, iy);
    }

#ifdef DEBUG_DRAW
    // Draw final points
    for (const auto& p : GDC_Adaptive_Grid_Points)
    {
        circle(debugImage, p, 2, Scalar(0, 0, 255), -1);
    }
    imshow("Adaptive Grid (Center-of-HighDistortion-Tile)", debugImage);
    waitKey();
#endif
}

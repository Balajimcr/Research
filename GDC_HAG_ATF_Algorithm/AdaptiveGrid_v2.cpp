#include "AdaptiveGrid_v2.h"
// Uncomment to enable debug drawing (requires display)
//#define DEBUG_DRAW

#ifdef DEBUG_DRAW
static void displayAndSaveImage(const cv::Mat& img, const std::string& winName) {
    cv::imshow(winName, img);
    // cv::imwrite(winName + ".png", img);  // If you wish to save
    cv::waitKey(1);
}
#endif

// Helper to clamp values
static inline int clampInt(int val, int minVal, int maxVal) {
    return std::max(minVal, std::min(maxVal, val));
}

// ---------------------------------------------------------------------------
// Weighted Voronoi + Adaptive Grid Generation
// ---------------------------------------------------------------------------
static void GenerateAdaptiveGrid_v2(
    const cv::Mat& magnitude_of_distortion,
    std::vector<cv::Point>& GDC_Adaptive_Grid_Points,
    const int targetGridX,
    const int targetGridY,
    const float LowThreshold
) {
#ifdef DEBUG_DRAW
    cv::Mat debugImage;
    if (magnitude_of_distortion.type() == CV_32F) {
        double minV, maxV;
        cv::minMaxLoc(magnitude_of_distortion, &minV, &maxV);
        // Normalize for visualization
        magnitude_of_distortion.convertTo(debugImage, CV_8UC1, 255.0 / (maxV - minV), -255.0 * minV / (maxV - minV));
        cv::cvtColor(debugImage, debugImage, cv::COLOR_GRAY2BGR);
    }
    else {
        // If already 8-bit or 3-channel
        debugImage = magnitude_of_distortion.clone();
        if (debugImage.channels() == 1) {
            cv::cvtColor(debugImage, debugImage, cv::COLOR_GRAY2BGR);
        }
    }
#endif

    // Basic checks
    CV_Assert(!magnitude_of_distortion.empty() && magnitude_of_distortion.type() == CV_32F);
    const int width = magnitude_of_distortion.cols;
    const int height = magnitude_of_distortion.rows;

    // Clear output vector
    GDC_Adaptive_Grid_Points.clear();

    // -----------------------------------------------------------------------
    // (1) INITIAL POINTS (could be random or a coarse grid). 
    //     We will assume a coarse grid initialization.
    // -----------------------------------------------------------------------
    // For Weighted Voronoi, each point has a weight. 
    // We set the weight ~ distortion value to concentrate more in high-distortion areas.
    // For demonstration, we place an initial coarse grid and sample the magnitude as weight.
    std::vector<cv::Point2f> initialPoints;
    std::vector<float>       initialWeights;  // same size as initialPoints

    const int initGridX = 10; // or any small number for initial seeding
    const int initGridY = 10;

    float stepX = (float)width / (float)(initGridX);
    float stepY = (float)height / (float)(initGridY);

    for (int gx = 0; gx < initGridX; gx++) {
        for (int gy = 0; gy < initGridY; gy++) {
            float px = (gx + 0.5f) * stepX;
            float py = (gy + 0.5f) * stepY;
            int ix = clampInt((int)std::floor(px), 0, width - 1);
            int iy = clampInt((int)std::floor(py), 0, height - 1);

            float distortionVal = magnitude_of_distortion.at<float>(iy, ix);
            // Scale or clamp distortionVal if needed
            // For Weighted Voronoi, we set weight ~ distortion + some epsilon to avoid zero weights
            float w = distortionVal + 0.001f;

            initialPoints.push_back(cv::Point2f(px, py));
            initialWeights.push_back(w);
        }
    }

#ifdef USE_CGAL
    // -----------------------------------------------------------------------
    // (2) BUILD WEIGHTED VORONOI DIAGRAM (POWER DIAGRAM) USING CGAL
    // -----------------------------------------------------------------------
    typedef CGAL::Exact_predicates_inexact_constructions_kernel                K;
    typedef CGAL::Regular_triangulation_euclidean_traits_2<K>                  Traits;
    typedef CGAL::Regular_triangulation_2<Traits>                              RT;
    typedef Traits::Point_2                                                    Point_2;
    typedef Traits::Weighted_point_2                                           Weighted_point_2;

    // Insert the initial set of Weighted points
    std::vector<Weighted_point_2> cgalPoints;
    cgalPoints.reserve(initialPoints.size());

    for (size_t i = 0; i < initialPoints.size(); i++) {
        float px = initialPoints[i].x;
        float py = initialPoints[i].y;
        float w = initialWeights[i];
        cgalPoints.push_back(Weighted_point_2(Point_2(px, py), w));
    }

    // Create the regular triangulation
    RT rt;
    rt.insert(cgalPoints.begin(), cgalPoints.end());

    // (2a) Extract the Voronoi cells & compute their centroids
    //      For each finite face in the triangulation, get the dual (Voronoi cell).
    std::vector<cv::Point2f> samplePoints;
    samplePoints.reserve(rt.number_of_faces());

    // Loop over finite faces
    for (auto fit = rt.finite_faces_begin(); fit != rt.finite_faces_end(); ++fit) {
        // Each face corresponds to a region in the power diagram. 
        // We can approximate the centroid by sampling or computing polygon centroid directly.

        // For simplicity, just use the face's circumcenter or face center as a placeholder
        // (This is not a rigorous centroid of the cell but is often used as a starting approach)
        K::Point_2 cc = rt.dual(fit);
        cv::Point2f cPt((float)cc.x(), (float)cc.y());

        // Make sure we clamp to image boundaries
        cPt.x = std::max(0.0f, std::min((float)width - 1, cPt.x));
        cPt.y = std::max(0.0f, std::min((float)height - 1, cPt.y));

        samplePoints.push_back(cPt);
    }

#else
    // -----------------------------------------------------------------------
    // (Fallback) If CGAL is not available, just treat the initial points 
    // as if they were derived from a Weighted Voronoi. 
    // In practice, you'd want to implement your own or use another library.
    // -----------------------------------------------------------------------
    std::vector<cv::Point2f> samplePoints = initialPoints;
#endif // USE_CGAL

    // -----------------------------------------------------------------------
    // (3) ADAPTIVE POINT REFINEMENT to reach EXACTLY targetGridX * targetGridY points
    // -----------------------------------------------------------------------
    // We'll call the final count = N*N:
    const int desiredCount = targetGridX * targetGridY;

    // Helper lambda to compute a rough "weighted area" for each point's region
    auto computeWeightedArea = [&](const cv::Point2f& pt) -> float {
        // Very naive approach: sample local area around pt
        // In a real scenario, you'd integrate distortion over the Voronoi cell area.
        // For demonstration, let's just read the distortion at the nearest pixel.
        int ix = clampInt((int)std::round(pt.x), 0, width - 1);
        int iy = clampInt((int)std::round(pt.y), 0, height - 1);
        return magnitude_of_distortion.at<float>(iy, ix);
        };

    // Build an initial list of points
    std::vector<cv::Point2f> currentPoints = samplePoints;

    // Oversampling or Undersampling
    // We will do repeated merges (if we have too many) or splits (if we have too few).
    while ((int)currentPoints.size() != desiredCount) {
        if ((int)currentPoints.size() < desiredCount) {
            // O V E R S A M P L I N G
            // 1) Find the point (cell) with the largest weighted area
            float maxArea = -1.0f;
            int   maxIdx = -1;
            for (size_t i = 0; i < currentPoints.size(); i++) {
                float area = computeWeightedArea(currentPoints[i]);
                if (area > maxArea) {
                    maxArea = area;
                    maxIdx = (int)i;
                }
            }
            if (maxIdx >= 0) {
                // 2) Place a new point near the local maximum of distortion 
                //    in the cell region. For simplicity, just place it offset by a small random
                //    or near the same location if we had more advanced search we'd do it properly.
                cv::Point2f maxPt = currentPoints[maxIdx];
                // Jitter or local max search (dummy approach):
                float offset = 2.0f; // small offset
                float nx = maxPt.x + offset * 0.5f;
                float ny = maxPt.y + offset * 0.5f;
                nx = std::max(0.0f, std::min((float)width - 1, nx));
                ny = std::max(0.0f, std::min((float)height - 1, ny));

                currentPoints.push_back(cv::Point2f(nx, ny));
            }
        }
        else {
            // U N D E R S A M P L I N G
            // 1) Find the point (cell) with the smallest weighted area
            float minArea = 9999999.0f;
            int   minIdx = -1;
            for (size_t i = 0; i < currentPoints.size(); i++) {
                float area = computeWeightedArea(currentPoints[i]);
                if (area < minArea) {
                    minArea = area;
                    minIdx = (int)i;
                }
            }
            if (minIdx >= 0 && currentPoints.size() > 1) {
                // 2) Remove the point that is closest to another point in that "cell".
                //    For demonstration, we remove minIdx itself or the one that’s 
                //    extremely close to it.
                currentPoints.erase(currentPoints.begin() + minIdx);
            }
        }
    }

    // -----------------------------------------------------------------------
    // (4) LLOYD'S RELAXATION (OPTIONAL)
    // -----------------------------------------------------------------------
    // The idea: Recompute Voronoi cells of currentPoints, recenter each point 
    // to centroid of its cell. We'll do a few iterations of that.
    // This step can be repeated as needed.
#ifdef USE_CGAL
    {
        const int lloydIterations = 3; // tune as needed
        for (int iter = 0; iter < lloydIterations; iter++) {
            // Build Weighted Voronoi with currentPoints
            // For simplicity, treat each point's weight as average distortion
            // or something stable. We skip weighting here and do standard Lloyd on unweighted.

            typedef CGAL::Delaunay_triangulation_2<K> Delaunay;
            std::vector<K::Point_2> dtPoints;
            dtPoints.reserve(currentPoints.size());
            for (auto& cp : currentPoints) {
                dtPoints.push_back(K::Point_2(cp.x, cp.y));
            }
            Delaunay dt;
            dt.insert(dtPoints.begin(), dtPoints.end());

            // Recompute each cell's centroid
            std::vector<cv::Point2f> newPositions;
            newPositions.resize(currentPoints.size(), cv::Point2f(0, 0));
            std::vector<int> counts(currentPoints.size(), 0);

            // Map from vertex to index
            std::map<K::Point_2, int> pointIndex;
            for (size_t i = 0; i < dtPoints.size(); i++) {
                pointIndex[dtPoints[i]] = (int)i;
            }

            // We can approximate the centroid by averaging sample positions 
            // within each cell, or do polygon-based exact centroid.
            // For demonstration, let's just average the adjacent triangles 
            // for which a point is a vertex:
            for (auto fit = dt.finite_faces_begin(); fit != dt.finite_faces_end(); ++fit) {
                // We get the center (e.g., triangle centroid)
                K::Point_2 p1 = fit->vertex(0)->point();
                K::Point_2 p2 = fit->vertex(1)->point();
                K::Point_2 p3 = fit->vertex(2)->point();
                float cx = (float)(p1.x() + p2.x() + p3.x()) / 3.0f;
                float cy = (float)(p1.y() + p2.y() + p3.y()) / 3.0f;

                // Accumulate to each vertex
                for (int v = 0; v < 3; v++) {
                    auto it = pointIndex.find(fit->vertex(v)->point());
                    if (it != pointIndex.end()) {
                        newPositions[it->second].x += cx;
                        newPositions[it->second].y += cy;
                        counts[it->second]++;
                    }
                }
            }

            // Move each point to the average of its centroid accumulation
            for (size_t i = 0; i < currentPoints.size(); i++) {
                if (counts[i] > 0) {
                    newPositions[i].x /= (float)counts[i];
                    newPositions[i].y /= (float)counts[i];
                }
                // clamp
                newPositions[i].x = std::max(0.0f, std::min((float)width - 1, newPositions[i].x));
                newPositions[i].y = std::max(0.0f, std::min((float)height - 1, newPositions[i].y));
            }

            // update
            for (size_t i = 0; i < currentPoints.size(); i++) {
                currentPoints[i] = newPositions[i];
            }
        }
    }
#endif // USE_CGAL

    // -----------------------------------------------------------------------
    // (5) FINAL OUTPUT
    // -----------------------------------------------------------------------
    // Convert cv::Point2f to cv::Point (int) for final usage
    GDC_Adaptive_Grid_Points.reserve(currentPoints.size());
    for (auto& cp : currentPoints) {
        int x = clampInt((int)std::round(cp.x), 0, width - 1);
        int y = clampInt((int)std::round(cp.y), 0, height - 1);
        GDC_Adaptive_Grid_Points.push_back(cv::Point(x, y));
    }

#ifdef DEBUG_DRAW
    // Visualize final points
    for (auto& p : GDC_Adaptive_Grid_Points) {
        cv::circle(debugImage, p, 2, cv::Scalar(0, 0, 255), -1);
    }
    displayAndSaveImage(debugImage, "Adaptive_Grid_Points_WeightedVoronoi");
    cv::waitKey();
#endif
}

// Uncomment to enable debug visualization
//#define DEBUG_DRAW

using namespace std;
using namespace cv;

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
 * @brief Reassign pixels to the closest seed (standard nearest-seed),
 *        forming clusters. Returns vector of clusters: clusters[i] are
 *        the points belonging to seeds[i].
 */
static vector<vector<Point>> segmentByNearestSeed(
    const Mat& distortionMap,
    const vector<Point2f>& seeds)
{
    const int rows = distortionMap.rows;
    const int cols = distortionMap.cols;

    vector<vector<Point>> clusters(seeds.size());
    clusters.reserve(seeds.size());

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            float distMin = FLT_MAX;
            int   bestIdx = 0;
            for (size_t s = 0; s < seeds.size(); ++s)
            {
                float dx = x - seeds[s].x;
                float dy = y - seeds[s].y;
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
 * @brief Finds the point of maximum distortion (>= LowThreshold) in a cluster.
 *        Returns a boolean indicating success/failure.
 */
static bool findLocalMaxDistortion(
    const Mat& distortionMap,
    const vector<Point>& clusterPixels,
    float LowThreshold,
    Point& outMaxPt)
{
    float localMaxVal = -1.0f;
    bool found = false;
    for (const auto& px : clusterPixels)
    {
        float val = distortionMap.at<float>(px.y, px.x);
        if (val >= LowThreshold && val > localMaxVal)
        {
            localMaxVal = val;
            outMaxPt = px;
            found = true;
        }
    }
    return found;
}

/**
 * @brief Oversample by adding new seeds in *all* clusters whose total distortion
 *        (above LowThreshold) exceeds someAreaThreshold. This process repeats
 *        in multiple iterations until we have at least targetCount seeds or no
 *        more clusters are above threshold.
 *
 * @param distortionMap  - input map [0..1]
 * @param seeds          - (in/out) current set of seeds
 * @param targetCount    - desired # of seeds
 * @param LowThreshold   - below this value, ignore distortion
 * @param someAreaThreshold - minimal total cluster distortion to justify adding another seed
 * @param maxIterations  - safety limit on how many oversample iterations to run
 */
static void oversamplePointsInAllHighDistortionClusters(
    const Mat& distortionMap,
    vector<Point2f>& seeds,
    int    targetCount,
    float  LowThreshold,
    float  someAreaThreshold,
    int    maxIterations = 10
)
{
    RNG rng(getTickCount());

    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        if ((int)seeds.size() >= targetCount)
            break; // we're done or already above target

        // 1) Segment the image by nearest seed
        vector<vector<Point>> clusters = segmentByNearestSeed(distortionMap, seeds);

        // 2) Compute weighted area (above LowThreshold)
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

        // 3) Attempt to add new seeds in all clusters that exceed someAreaThreshold
        bool anyClusterAdded = false;
        for (size_t c = 0; c < clusters.size(); ++c)
        {
            if ((int)seeds.size() >= targetCount)
                break; // reached desired number

            if (clusterWeightedArea[c] > someAreaThreshold)
            {
                // Find local maximum distortion in that cluster
                Point localMaxPt;
                bool foundMax = findLocalMaxDistortion(distortionMap, clusters[c], LowThreshold, localMaxPt);
                if (foundMax)
                {
                    // Add a small random offset to avoid duplicates
                    float jitterX = rng.uniform(-1.f, 1.f);
                    float jitterY = rng.uniform(-1.f, 1.f);
                    Point2f newSeed(localMaxPt.x + jitterX, localMaxPt.y + jitterY);
                    newSeed = clampPoint2f(newSeed, (float)distortionMap.cols, (float)distortionMap.rows);

                    seeds.push_back(newSeed);
                    anyClusterAdded = true;
                }
            }
        }

        // If we didn't add any new seeds in this iteration, no point continuing
        if (!anyClusterAdded)
            break;
    }
}

/** Weighted Voronoi / Lloyd-like relaxation (for reference) */
static void approximateWeightedVoronoi(
    const Mat& distortionMap,
    vector<Point2f>& seeds,
    int   iterations,
    float alpha,
    float LowThreshold)
{
    const int width = distortionMap.cols;
    const int height = distortionMap.rows;

    for (int iter = 0; iter < iterations; ++iter)
    {
        // Create clusters for each seed
        vector<vector<Point>> clusters(seeds.size());

        // 1) Assign each pixel to the closest seed (weighted distance)
        for (int y = 0; y < height; ++y)
        {
            const float* dptr = distortionMap.ptr<float>(y);
            for (int x = 0; x < width; ++x)
            {
                // Weighted value (0 if below threshold)
                float weightVal = (dptr[x] < LowThreshold) ? 0.f : dptr[x];

                float distMin = FLT_MAX;
                int   bestIdx = 0;
                for (size_t s = 0; s < seeds.size(); ++s)
                {
                    float dx = x - seeds[s].x;
                    float dy = y - seeds[s].y;
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
        for (size_t s = 0; s < seeds.size(); ++s)
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
                    seeds[s].x = (float)(sumX / wSum);
                    seeds[s].y = (float)(sumY / wSum);
                }
                // clamp
                seeds[s] = clampPoint2f(seeds[s], (float)width, (float)height);
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
    // We want exactly Grid_x * Grid_y points
    const int targetCount = Grid_x * Grid_y;

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
    // 1) Initialize seeds (uniformly)
    //--------------------------------------------------------------------------
    vector<Point2f> seeds;
    seeds.reserve(targetCount);
    for (int i = 0; i < Grid_x; i++)
    {
        for (int j = 0; j < Grid_y; j++)
        {
            float x = (float)i * (width - 1) / (float)(Grid_x - 1);
            float y = (float)j * (height - 1) / (float)(Grid_y - 1);
            seeds.emplace_back(x, y);
        }
    }

    //--------------------------------------------------------------------------
    // 2) Weighted Voronoi relaxation (optional)
    //--------------------------------------------------------------------------
    approximateWeightedVoronoi(magnitude_of_distortion,
        seeds,
        /*iterations=*/3,
        /*alpha=*/10.f,
        LowThreshold);

    //--------------------------------------------------------------------------
    // 3) Oversample in ALL high-distortion clusters
    //    - This function will repeatedly add seeds in high-distortion clusters
    //      until we have at least 'targetCount' or no more clusters exceed threshold.
    //--------------------------------------------------------------------------
    float someAreaThreshold = 0.0;
    // ^ This area threshold is separate from LowThreshold. 
    //   If your distortionMap is (0..1), you might use a smaller or bigger value 
    //   depending on the image size & distribution.

    oversamplePointsInAllHighDistortionClusters(
        magnitude_of_distortion,
        seeds,
        targetCount,
        LowThreshold,
        someAreaThreshold,
        /*maxIterations=*/10
    );

    //--------------------------------------------------------------------------
    // 4) If we exceeded the targetCount, you may do an optional "undersample" pass
    //    to remove seeds that are too close or in low-distortion clusters, etc.
    //    For brevity, not shown here. 
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // 5) Convert float seeds to integer points
    //--------------------------------------------------------------------------
    GDC_Adaptive_Grid_Points.clear();
    GDC_Adaptive_Grid_Points.reserve(seeds.size());
    for (const auto& s : seeds)
    {
        int ix = clampValue((int)std::round(s.x), 0, width - 1);
        int iy = clampValue((int)std::round(s.y), 0, height - 1);
        GDC_Adaptive_Grid_Points.emplace_back(ix, iy);
    }

#ifdef DEBUG_DRAW
    // Visualization
    for (const auto& p : GDC_Adaptive_Grid_Points)
    {
        circle(debugImage, p, 2, Scalar(0, 0, 255), -1);
    }
    imshow("Adaptive Grid (Oversample All High Distortion)", debugImage);
    waitKey();
#endif
}
#pragma once
#include <opencv2/opencv.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>




// ---------------------------- CGAL Includes (Weighted Voronoi) ----------------------------
#ifdef USE_CGAL
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Regular_triangulation_euclidean_traits_2.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/Polygon_2.h>
#endif
// -------------------------------------------------------------------------------------------



static void GenerateAdaptiveGrid_v2(
    const cv::Mat& magnitude_of_distortion,
    std::vector<cv::Point>& GDC_Adaptive_Grid_Points,
    const int targetGridX,
    const int targetGridY,
    const float LowThreshold
);

void GenerateAdaptiveGrid_HAG_WeightedVoronoi(
    const cv::Mat& magnitude_of_distortion,
    std::vector<cv::Point>& GDC_Adaptive_Grid_Points,
    const int Grid_x,
    const int Grid_y,
    const float LowThreshold
);


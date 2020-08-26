/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template code for
 *	the assignment LAB 4 "Histogram-based tracking"
 *
 *	Header of utilities for LAB4.
 *	Some of these functions are adapted from OpenSource
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <string> 		// for string class
#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<cv::Rect> readGroundTruthFile(std::string groundtruth_path);
std::vector<float> estimateTrackingPerformance(std::vector<cv::Rect> Bbox_GT, std::vector<cv::Rect> Bbox_est);
Mat returnHistogramAroundRegion(Mat cropped_frame, int hist_size);
cv::Mat setHistogramColor(Mat frame, char color_features);
void calculateCandidates(Mat singleChannelFrame,
		std::vector<Mat> &candidates,
		std::vector<Rect> &candidate_rectangle,
		Rect currentState,
		int size_of_neighbourhood,
		int step_size,
		int hist_size
		);

cv::Mat histogramImage(Mat  basicModelHistogram,
		Mat currentHistogram,
		int hist_size);
int selectBestCandidate(Mat basicModelHistogram, std::vector<Mat> candidates);
#endif /* UTILS_HPP_ */

/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template code for
 *	the assignment LAB 4 "Histogram-based tracking"
 *
 *	Implementation of utilities for LAB4.
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <vector>

using namespace cv;
using namespace std;

/**
 * Reads a text file where each row contains comma separated values of
 * corners of groundtruth bounding boxes.
 *
 * The annotations are stored in a text file with the format:
 * Row format is "X1, Y1, X2, Y2, X3, Y3, X4, Y4" where Xi and Yi are
 * the coordinates of corner i of the bounding box in frame N, which
 * corresponds to the N-th row in the text file.
 *
 * Returns a list of cv::Rect with the bounding boxes data.
 *
 * @param ground_truth_path: full path to ground truth text file
 * @return bbox_list: list of ground truth bounding boxes of class Rect
 */
std::vector<Rect> readGroundTruthFile(std::string groundtruth_path)
{
	// variables for reading text file
	ifstream inFile; //file stream
	string bbox_values; //line of file containing all bounding box data
	string bbox_value;  //a single value of bbox_values

	vector<Rect> bbox_list; //output with all read bounding boxes

	// open text file
	inFile.open(groundtruth_path.c_str(),ifstream::in);
	if(!inFile)
		throw runtime_error("Could not open groundtrutfile " + groundtruth_path); //throw error if not possible to read file

	// Read each line of groundtruth file
	while(getline(inFile, bbox_values)){

		stringstream linestream(bbox_values); //convert read line to linestream
		//cout << "-->lineread=" << linestream.str() << endl;

		// Read comma separated values of groundtruth.txt
		vector<int> x_values,y_values; 	//values to be read from line
		int line_ctr = 0;						//control variable to read alternate Xi,Yi
		while(getline(linestream, bbox_value, ',')){

			//read alternate Xi,Yi coordinates
			if(line_ctr%2 == 0)
				x_values.push_back(stoi(bbox_value));
			else
				y_values.push_back(stoi(bbox_value));
			line_ctr++;
		}

		// Get width and height; and minimum X,Y coordinates
		double xmin = *min_element(x_values.begin(), x_values.end()); //x coordinate of the top-left corner
		double ymin = *min_element(y_values.begin(), y_values.end()); //y coordinate of the top-left corner

		if (xmin < 0) xmin=0;
		if (ymin < 0) ymin=0;

		double width = *max_element(x_values.begin(), x_values.end()) - xmin; //width
		double height = *max_element(y_values.begin(), y_values.end()) - ymin;//height

		// Initialize a cv::Rect for a bounding box and store it in a std<vector> list
		bbox_list.push_back(Rect(xmin, ymin, width, height));
		//std::cout << "-->Bbox=" << bbox_list[bbox_list.size()-1] << std::endl;
	}
	inFile.close();

	return bbox_list;
}

/**
 * Compare two lists of bounding boxes to estimate their overlap
 * using the criterion IOU (Intersection Over Union), which ranges
 * from 0 (worst) to 1(best) as described in the following paper:
 * Čehovin, L., Leonardis, A., & Kristan, M. (2016).
 * Visual object tracking performance measures revisited.
 * IEEE Transactions on Image Processing, 25(3), 1261-1274.
 *
 * Returns a list of floats with the IOU for each frame.
 *
 * @param Bbox_GT: list of elements of type cv::Rect describing
 * 				   the groundtruth bounding box of the object for each frame.
 * @param Bbox_est: list of elements of type cv::Rect describing
 * 				   the estimated bounding box of the object for each frame.
 * @return score: list of float values (IOU values) for each frame
 *
 * Comments:
 * 		- The two lists of bounding boxes must be aligned, meaning that
 * 		position 'i' for both lists corresponds to frame 'i'.
 * 		- Only estimated Bboxes are compared, so groundtruth Bbox can be
 * 		a list larger than the list of estimated Bboxes.
 */
std::vector<float> estimateTrackingPerformance(std::vector<cv::Rect> Bbox_GT, std::vector<cv::Rect> Bbox_est)
{
	vector<float> score;

	//For each data, we compute the IOU criteria for all estimations
	for(int f=0;f<(int)Bbox_est.size();f++)
	{
		Rect m_inter = Bbox_GT[f] & Bbox_est[f];//Intersection
		Rect m_union = Bbox_GT[f] | Bbox_est[f];//Union

		score.push_back((float)m_inter.area()/(float)m_union.area());
	}

	return score;
}

//Caculates the histogram of the image
//cropped_frame: image, we want to get the histogram of
//hist_size: the number of bins of the histogram
//returns: the histogram of the region
cv::Mat returnHistogramAroundRegion(Mat cropped_frame, int hist_size){
	Mat tempHistogram;
	Mat basicModelHistogram;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	calcHist(&cropped_frame, 1, 0, Mat(), tempHistogram, 1, &hist_size, &histRange, true, false);
	normalize(tempHistogram, basicModelHistogram, 0, 1, NORM_MINMAX, -1, Mat() );

	return basicModelHistogram;
}

//Transforms the image into the desierd color channel
//frame: the image we wanna transform
//color_features: charcode of the color
//H: H channel HSV
//S: S channel HSV
//R: R channel RGB
//G: G channel RGB
//B: B channel RGB
//everything else (default): gray_level
//returns: the color channel of the image
cv::Mat setHistogramColor(Mat frame, char color_features){
	Mat grayImage;
	vector<Mat> spl;
	Mat singleChannelFrame;
	switch (color_features){
	case 'H':
		cv::cvtColor(frame, grayImage, COLOR_BGR2HSV);
		split(grayImage,spl);
		singleChannelFrame = spl[0];
		break;
	case 'S':
			cv::cvtColor(frame, grayImage, COLOR_BGR2HSV);
			split(grayImage,spl);
			singleChannelFrame = spl[1];
			break;
	case 'R':
			split(frame,spl);
			singleChannelFrame = spl[2];
			break;
	case 'G':
			split(frame,spl);
			singleChannelFrame = spl[1];
			break;
	case 'B':
			split(frame,spl);
			singleChannelFrame = spl[0];
			break;
	default:
		cv::cvtColor(frame, grayImage, COLOR_BGR2GRAY);
		split(grayImage,spl);
		singleChannelFrame = spl[0];
	}
	return singleChannelFrame;
}


//Calculates the possible candidates in a neighbourhood
//singleChannelFrame: the
//candidates: An empty container where we save the histogram of candidates, will be filled in the function
//candidate_rectangle: An empty container where we save the location of candidates, will be filled in the function
//currentState: the (x,y) coordinates of the previous detected object
//size_of_neighbourhood: how far neighbours we wanna look in each direction
//step_size: how many pixels would be inspected
//hist_size number of bins in the histogram
void calculateCandidates(Mat singleChannelFrame,
		std::vector<Mat> &candidates,
		std::vector<Rect> &candidate_rectangle,
		Rect currentState,
		int size_of_neighbourhood,
		int step_size,
		int hist_size
		){

	int startx = currentState.x;
	int endx = currentState.x;
	int starty = currentState.y;
	int endy = currentState.y;
	//check if our window wont be out of the frame
	if(currentState.x-size_of_neighbourhood > 0){
		startx = currentState.x-size_of_neighbourhood;
	}else{
		startx = 0;
	}
	if(currentState.x+size_of_neighbourhood < singleChannelFrame.rows){
		endx = currentState.x+size_of_neighbourhood;
	}else{
		endx = singleChannelFrame.rows;
	}

	if(currentState.y-size_of_neighbourhood > 0){
		starty = currentState.y-size_of_neighbourhood;
	}else{
		starty = 0;
	}
	if(currentState.y+size_of_neighbourhood < singleChannelFrame.cols){
		endy = currentState.y+size_of_neighbourhood;
	}else{
		endy = singleChannelFrame.cols-1;
	}

	//iterate over all the possible candidate places, where i and j are the coordinates of the centers of the potential candidates
	for(int i = startx; i < endx; i+=step_size ){
		for(int j = starty; j < endy; j+=step_size){

			if( i > 0 && (i + currentState.width) <  singleChannelFrame.rows && j > 0 && (j + currentState.width) <  singleChannelFrame.cols  ){
				Mat singleCandidate;
				Mat cropped;
				//frame.copyTo(cropped);
				cropped = singleChannelFrame(Rect(i,j, currentState.width, currentState.height ));
				singleCandidate =returnHistogramAroundRegion(cropped, hist_size);
				candidates.push_back(singleCandidate);
				candidate_rectangle.push_back(Rect(i,j,currentState.width,currentState.height));
			}
		}
	}
}

//To visualize the model histogram, and histogram of the closest candidate
//basicModelHistogram is the model, it will be shown red
//currentHistogram is the best candidate's histogram, it will be shown green
//hist_size number of bins in the histogram
//returns: the image with both plotted histogram
cv::Mat histogramImage(Mat  basicModelHistogram,
		Mat currentHistogram,
		int hist_size){

	Mat winnerHistogram;
	Mat modelNormalizedHistogram;

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/hist_size );
	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	// Normalize the result to [ 0, histImage.rows ]
	normalize(basicModelHistogram, modelNormalizedHistogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(currentHistogram, winnerHistogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	// Draw for each channel
	for( int i = 1; i < hist_size; i++ )
	{
	line( histImage, Point( bin_w*(i-1), hist_h - cvRound(winnerHistogram.at<float>(i-1)) ) ,
			   Point( bin_w*(i), hist_h - cvRound(winnerHistogram.at<float>(i)) ),
			   Scalar( 0, 255, 0), 2, 8, 0  );
	line( histImage, Point( bin_w*(i-1), hist_h - cvRound(modelNormalizedHistogram.at<float>(i-1)) ) ,
						   Point( bin_w*(i), hist_h - cvRound(modelNormalizedHistogram.at<float>(i)) ),
						   Scalar( 0, 0, 255), 2, 8, 0  );
	}
	//namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	//imshow("calcHist Demo", histImage );
	return histImage;
	//waitKey(0);
}

//We select the best histogram based on Battacharyya distance
//the inputs are the histogram of the model, and a vector containing the list of candidate histograms
//return the index of the best matching candidate
int selectBestCandidate(Mat basicModelHistogram, std::vector<Mat> candidates){
	//Compare object and its candidates:
	int index_with_smallest_distance = 0;
	double smallest_dist = compareHist(basicModelHistogram, candidates[0], CV_COMP_BHATTACHARYYA);
	for(int i = 1; i < candidates.size(); i++){
		double current_distance = compareHist(basicModelHistogram, candidates[i], CV_COMP_BHATTACHARYYA);
		if(current_distance < smallest_dist){
			smallest_dist = current_distance;
			index_with_smallest_distance = i;
		}
	}

	return index_with_smallest_distance;
}



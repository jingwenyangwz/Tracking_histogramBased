/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template (sample program)
 *	provided for the assignment LAB 4 "Histogram-based tracking"
 *
 *	This code has been tested using:
 *	- Operative System: Ubuntu 18.04
 *	- OpenCV version: 3.4.4
 *	- Eclipse version: 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
//includes
#include <stdio.h> 								//Standard I/O library
#include <numeric>								//For std::accumulate function
#include <string> 								//For std::to_string function
#include <opencv2/opencv.hpp>					//opencv libraries
#include "utils.hpp" 							//for functions readGroundTruthFile & estimateTrackingPerformance
#include <math.h>
#include "ShowManyImages.hpp"

//namespaces
using namespace cv;
using namespace std;
/*
	float szamlalo = 0;
	char color_features = 'd';
	int allcandidatesize = 0;
	std::vector<float> scores;
	int hist_size = 16;
	*/
//main function
int main(int argc, char ** argv)
{

	//PLEASE CHANGE 'dataset_path' & 'output_path' ACCORDING TO YOUR PROJECT
	std::string dataset_path = "/home/peter/eclipse-workspace/Ex1_LoadModifySave/datasets";									//dataset location.
	std::string output_path = "/home/peter/eclipse-workspace/Ex1_LoadModifySave/outvideos/";										//location to save output videos

	// dataset paths
	/*std::string sequences[] = {"bag","ball","road",};*/					//test data for lab4.6
	//std::string sequences[] = {"bag"};
	//std::string sequences[] = {"ball"};
	std::string sequences[] = {"bag"};
	std::string image_path = "%08d.jpg"; 									//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 						//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);					//number of sequences

	//Loop for all sequence of each category
	for (int s=0; s<NumSeq; s++ )
	{
		Mat frame;										//current Frame
		int frame_idx=0;								//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;	//estimated & groundtruth bounding boxes
		std::vector<double> procTimes;					//vector to accumulate processing times

		std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path; //path of videofile. DO NOT CHANGE
		VideoCapture cap(inputvideo);	// reader to grab frames from videofile

		//check if videofile exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); //error if not possible to read videofile

		// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));//cv::Size frame_size(700,460);
		VideoWriter outputvideo(output_path+"outvid_" + sequences[s]+".avi",CV_FOURCC('X','V','I','D'),10, frame_size);	//xvid compression (cannot be changed in OpenCV)

		//Read ground truth file and store bounding boxes
		std::string inputGroundtruth = dataset_path + "/" + sequences[s] + "/" + groundtruth_file;//path of groundtruth file. DO NOT CHANGE
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); //read groundtruth bounding boxes

		//main loop for the sequence
		std::cout << "Displaying sequence at " << inputvideo << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;

		Rect objectModel = Rect(-1, -1, 0, 0);
		Rect currentState;

		// MODEL SELECTION,
		// only_color: true  + only_gradient: false -> color model
		// only_color: false + only_gradient: true  -> gradient model
		// only_color: false + only_gradient: false -> combined model
		bool only_color = false;
		bool only_gradient = false;

		//Color histogram parameter
		char color_features = 'G'; // here we discover the color channel in GRAY, H, S, R, G, B;  GRAY works the best in task4.2
		int color_hist_size = 16; // we assume 16 bins for color histogram, FIXED
		//Gradient Hisogram parameters:

		int gradient_hist_size = 7; // set to 7bins according to task 4.4 sequence basketball

		// searching neighborhood radius and stride
		int size_of_neighbourhood = 36;
		int step_size = 1;// stride should be 1 or 2 according to taks4.4
		// number of candidates = (gradient_size_of_neighbourhood*2/gradient_step_size)^2
		// number of candidates around 225 with stride 2, around 900 with stride 1

		// initialize the target color/gradient histogram model
		Mat basicModelHistogram_color;
		Mat basicModelHistogram_gradient;
		Mat ColorModelImage, GradientModelImage;

		for (;;){

			/**/
			//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
			cap >> frame;

			if (!frame.data)
				break;

			//Time measurement
			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);			//get the current frame

			////////////////////////////////////////////////////////////////////////////////////////////
			//DO TRACKING
			//Change the following line with your own code
			//list_bbox_est.push_back(Rect(20,20,40,50));//we use a fixed value only for this demo program. Remove this line when you use your code
			//...
			// ADD YOUR CODE HERE
			// convert current frame into expected channel
			Mat singleChannelFrame = setHistogramColor(frame, color_features);
			Mat GrayScaleFrame = setHistogramColor(frame, 'GRAY');

			if(frame_idx == 1){
				//If first frame, use groundtruth to get x,y position of model
				// and to initialize the model
				objectModel.x = list_bbox_gt[frame_idx-1].x; //top left coordinate
				objectModel.y = list_bbox_gt[frame_idx-1].y; //top left coordinate
				objectModel.width = list_bbox_gt[frame_idx-1].width; //width and height will remain the same thoughout the video
				objectModel.height = list_bbox_gt[frame_idx-1].height;

				//get cropped image according to the boudning box
				ColorModelImage = singleChannelFrame(objectModel);
				GradientModelImage = GrayScaleFrame(objectModel);

				//And calculate the model (extract histograms)
				basicModelHistogram_color = returnColorHistogramAroundRegion(ColorModelImage, color_hist_size);
				basicModelHistogram_gradient = returnGradientHistogramAroundRegion(GradientModelImage, gradient_hist_size);
				currentState = objectModel;

			}else{
				//Get previous result center
				currentState.x = list_bbox_est[frame_idx-2].x;
				currentState.y = list_bbox_est[frame_idx-2].y;
			}

			//Calculate candidates:
			std::vector<Mat> candidates_color; //vector of the histogram of the candidates
			std::vector<Mat> candidates_gradient; //vector of the histogram of the candidates

			std::vector<Rect> candidate_rectangle_c; //vector of the candidate rectangles (position of candidates)
			//std::vector<Rect> candidate_rectangle_g; //vector of the candidate rectangles (position of candidates)
			std::vector<float> distances_color;
			std::vector<float> distances_gradient;
			int index_with_smallest_distance_combined;

			//calculate the candidate locations, and histograms
			calculateCandidates(singleChannelFrame, candidates_color, candidate_rectangle_c, currentState, size_of_neighbourhood, step_size, color_hist_size,true);
			calculateCandidates(singleChannelFrame, candidates_gradient, candidate_rectangle_c, currentState, size_of_neighbourhood, step_size, gradient_hist_size,false);

			//search for the best (closest) candidate
			CalculateDIstances(basicModelHistogram_color, candidates_color, distances_color, true);
			CalculateDIstances(basicModelHistogram_gradient, candidates_gradient, distances_gradient, false);
			index_with_smallest_distance_combined = selectCombinedCandidate(distances_color, distances_gradient,only_color,only_gradient);

			list_bbox_est.push_back(candidate_rectangle_c[index_with_smallest_distance_combined]);
			//visulaize the histograms
			cv::Mat plotColorHistograms = histogramImage(basicModelHistogram_color, candidates_color[index_with_smallest_distance_combined], color_hist_size);
			cv::Mat plotGradientHistograms = histogramImage(basicModelHistogram_gradient, candidates_gradient[index_with_smallest_distance_combined], gradient_hist_size);
			//...
			////////////////////////////////////////////////////////////////////////////////////////////

			//Time measurement
			procTimes.push_back(((double)getTickCount() - t)*1000. / cv::getTickFrequency());
			//std::cout << " processing time=" << procTimes[procTimes.size()-1] << " ms" << std::endl;

			// plot frame number & groundtruth bounding box for each frame
			putText(frame, std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)

			//show & save data
			//imshow("Tracking for "+sequences[s]+" (Green=GT, Red=Estimation)", frame);
			outputvideo.write(frame);//save frame to output video
			//and finally show our results: frame, candidate box (in RGB), model box (RGB) and both of the histograms
			ShowManyImages("Tracking", 6, frame, frame(list_bbox_est[frame_idx-1]),ColorModelImage,GradientModelImage, plotColorHistograms,plotGradientHistograms);

			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
			//waitKey();
			cout<<"frame NO. "<< frame_idx <<": number of candidates" << candidates_gradient.size()<<endl;
			vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);
			std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;

		}

		//comparison groundtruth & estimation
		vector<float> trackPerfALL = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);
		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerfALL.begin(), trackPerfALL.end(), 0.0) / trackPerfALL.size() << std::endl;
		waitKey(0);
		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
/*
	ofstream fajl;
	fajl.open("results.txt", std::ios_base::app);
	fajl << color_features;
	fajl<< "(";
	fajl << hist_size;
	fajl << ")\n";
	for(int i = 0; i < scores.size(); i++){
		fajl << sequences[i];
		fajl << ": ";
		fajl << allcandidatesize;
		fajl << ": ";
		fajl << scores[i];
		fajl << "\n";
	}
	fajl << "MEAN: ";
	fajl << szamlalo/2;
	fajl << "\n\n";
	fajl.close();

	printf("Finished program.");
	return 0;
	*/
}

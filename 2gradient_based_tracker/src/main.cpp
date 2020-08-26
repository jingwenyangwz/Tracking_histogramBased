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
#include <fstream>

//namespaces
using namespace cv;
using namespace std;

//main function
int main(int argc, char ** argv)
{
	//PLEASE CHANGE 'dataset_path' & 'output_path' ACCORDING TO YOUR PROJECT
	std::string dataset_path = "/home/peter/eclipse-workspace/Ex1_LoadModifySave/datasets";									//dataset location.
	std::string output_path = "/home/peter/eclipse-workspace/Ex1_LoadModifySave/outvideos/";									//location to save output videos

	// dataset paths
	/*std::string sequences[] = {"bolt1",										//test data for lab4.1, 4.3 & 4.5
							   "sphere","car1",								//test data for lab4.2
							   "ball2","basketball",						//test data for lab4.4
							   "bag","ball","road",};*/						//test data for lab4.6
	//std::string sequences[] = {"bolt1"};
	std::string sequences[] = {"car1","sphere"};
	std::string image_path = "%08d.jpg"; 									//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 						//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);					//number of sequences
	std::cout << NumSeq << std::endl;
	float szamlalo = 0;
	char color_features = 'd';
	int allcandidatesize = 0;
	std::vector<float> scores;
	int hist_size = 16;
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
		/*for(int i = 0; i < argc; i++){
			std::cout << i << " " <<argv[i] << std::endl;
		}*/

		int size_of_neighbourhood = 8;
		int step_size = 1;
		if( argc < 2 ){
			std::cout << "No arguments passed, values set to default" << std::endl;
		}else if (argc == 5){
			string valami = argv[1];
			color_features = valami[0];
			hist_size = std::stoi(argv[2]);
			size_of_neighbourhood = std::stoi(argv[3]);
			step_size = stoi(argv[4]);
		}
		std::cout << "Color Channel: " << color_features << std::endl;
		std::cout << "Number of bins: " << hist_size << std::endl;
		std::cout << "Size of neighbourhood set: " << size_of_neighbourhood << std::endl;
		std::cout << "Stride: " << step_size << std::endl;
		std::cout << "Number of candidates: ";
		//int size_of_neighbourhood = 12;
		//int step_size = 3;

		Mat basicModelHistogram;

		Mat modelImage;



		for (;;) {
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
			Mat singleChannelFrame = setHistogramColor(frame, color_features);

			if(frame_idx == 1){
				//If first frame, use groundtruth to get x,y position of model
				// and to initialize the model
				objectModel.x = list_bbox_gt[frame_idx-1].x; //top left coordinate
				objectModel.y = list_bbox_gt[frame_idx-1].y; //top left coordinate
				objectModel.width = list_bbox_gt[frame_idx-1].width; //width and height will remain the same thoughout the video
				objectModel.height = list_bbox_gt[frame_idx-1].height;

				modelImage = singleChannelFrame(objectModel);
				//And calculate the model (histogram)
				basicModelHistogram =returnHistogramAroundRegion(modelImage, hist_size);

				currentState = objectModel;
			}else{
				//Get previous result center
				currentState.x = list_bbox_est[frame_idx-2].x;
				currentState.y = list_bbox_est[frame_idx-2].y;
			}

			//Calculate candidates:
			std::vector<Mat> candidates; //vector of the histogram of the candidates
			std::vector<Rect> candidate_rectangle; //vector of the candidate rectangles (position of candidates)
			//calculate the candidate locations, and histograms
			calculateCandidates(singleChannelFrame, candidates, candidate_rectangle, currentState, size_of_neighbourhood, step_size, hist_size);
			if(frame_idx == 1){
				std::cout << candidates.size() << std::endl;
				allcandidatesize = candidates.size();
			}

			//search for the best (closest) candidate
			int index_with_smallest_distance = selectBestCandidate(basicModelHistogram, candidates);
			list_bbox_est.push_back(candidate_rectangle[index_with_smallest_distance]);
			//visulaize the histograms
			cv::Mat plotHistograms = histogramImage(basicModelHistogram, candidates[index_with_smallest_distance], hist_size);
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
			ShowManyImages("Tracking", 4, frame, plotHistograms, modelImage, frame(list_bbox_est[frame_idx-1]));

			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
			//waitKey();
		}

		//comparison groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);
		/*for(int i = 0; i< trackPerf.size(); i++){
			std::cout << trackPerf[i] << std::endl;
		}*/
		float avg = std::accumulate(trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size();
		std::cout<< "The MEAN: " << avg << std::endl;
		szamlalo += avg;
		scores.push_back(avg);
		//print stats about processing time and tracking performance
		//std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		//std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;
		//waitKey(0);
		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	//std::cout << allcandidatesize << std::endl;
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
	fajl << szamlalo/scores.size();
	fajl << "\n\n";
	fajl.close();

	std::cout << "AVG " << szamlalo/2 << std::endl;
	printf("Finished program.");
	return 0;
}

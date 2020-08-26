# Tracking_histogramBased
Single-object tracking based on histograms: color-based tracker, gradient-based tracker, and color-gradient fusion tracker



-task1:
  - Initialize an object model by using only the first frame and the groundtruth data.
  - Generate candidate locations using the grid approach described. (Focus only on object centers. Keep fixed the width/height of object location)
  - Compare object and candidate models using the Battacharyya distance
  - Select the candidate minimizing the **Battacharyya distance** (Eq1 w/o gradients) (please note that Eq1 maximizes similarity, here we will minimize distance)
  - parameters:
    - number of histograms bins (#bins) 
    - number of generated candidates (#cand)
  - evaluate the performance of the color-tracker with real sequence from [votchallengeNet](https://www.votchallenge.net/)


-task2:
  - Initialize an object model
  - Generate candidate locations
  - Compare object and candidate models using the **L2 distance**.
  - Select the candidate minimizing the L2 distance. (Eq 1 without the color score)
  - parameters:
    - number of bins (#bins) (#bins allows changing HOG descriptor length)
    - number of generated candidates (#cand)
    - evaluate the performance of the gradient-tracker with real sequence from [votchallengeNet](https://www.votchallenge.net/)

Modifications of the original paper:
  - As gradient features, compute the HOG descriptor using the HOGdescriptor class of OpenCV
  
  
  
-task3:
  - Initialize an object model 
  - Generate candidate locations using the grid approach described.
  - For each feature, obtain the scores by comparing object and candidate models. Then, obtain the normalized scores for each feature.
  - Select the candidate minimizing the combination of both normalized scores
  - The algorithm is able to choose each feature (color/gradient) or fusion.
  - evaluate the performance of the colorgradient_fusion_tracker with real sequence from [votchallengeNet](https://www.votchallenge.net/)
Modifications of the original paper:
  - As color features, use the color channel with highest performance in task 1
  - As gradient features, use HOG from task 2
  - As settings for both color and gradient features, use the number of candidates
and descriptor length based on your conclusions for task 1 and task 2.

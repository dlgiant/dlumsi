## Developing DL unsupervised workflows to identify Social Interaction between mice

### Summary: 
This project aims to define a DL workflow that can successfully label interactions between dyadic mice of the same gender, not necessarily same species. The distinction between mice is irrelevant to this study, the focus is at the relation between mice at any given time.

### Format: 
Videos are low resolution in gray scale, camera is fixed

### Tools: 
Python + OpenCV.

### Current step: 
Applying randomized projection in order to reduce the number of parameters decreasing the space used in network

### Sample frame from video:
![Mice interaction](/images/mice_interaction_example.png)

### Sample Lucas-Kanode optical flow feature extraction
![Lucas Kanode](/images/features_extracted_using_naive_lucas_kanode.png)


### Other optical flow feature extraction
![Other Optical Flow](/images/features_extracted_using_other_optical_flow.png)

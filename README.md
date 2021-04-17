# Artificial-IntelligenceProject
# CAR PLATE NUMBER RECOGNITION USING DEEP LEARNING. 

## A. PROJECT SUMMARY

**Project Title:** Face Mask Detection using Deep Learning

**Team Members:** 
-   NURIN SYAFIQAH BINTI MOHD SAOFI
- NURSYAFIQAH ADILLA BINTI MOHD SYAFIQ
- NOR AFIEQA BINTI MAHDI
- NUR FATIN BINTI YAZID


- [ ] **Objectives:**
- The main purpose of the project is 
- To utilize the usage of technology in road. 
- To decrease human work for detecting small number using bareyes.
- To minimal the number or road criminal and road accident. 


##  B. ABSTRACT 

Vechicle plate recognition has been studied since 1990s.The first approach was based on characteristic. The input image will be processed to enrich the boundry lines informtation by algorithms such as gradient filter and resulting in edging images. 

The image are process by certain algorithms.Hough Transform in detecting lines. Eventually, couple of two parallel lines will be considered as plate-candidate. The boundry line detection is not suitable of other than horizontal plate number. This will corrupting the boundry lines.The Hough transform inherently time-consuming. 

The colour and texture of the license plate can be used to identify but they seem to aimless and ineffective. 
As for now, we are going to use AI technology to recognize car plate registration number :  

![Coding](https://github.com/nursyafiqahJackson/Artificial-IntelligenceProject/blob/6e1c16d7ed3c9d5c3b4bf23188352611832c0409/download%20(2).jfif)


Figure 1 shows the AI output of plate number detection.


## C.  DATASET

In this project, we’ll discuss our four-phase vehicle number plate recognition , detailing how our computer vision/deep learning pipeline will be implemented.

From there, we’ll review the dataset we’ll be using to train our custom number plate recognition for Malaysia.

![Coding](https://github.com/nursyafiqahJackson/Artificial-IntelligenceProject/blob/main/download%20(5).jfif)
I’ll then show you how to implement a Python script to train a plate recognition on our dataset using Keras and TensorFlow.

We’ll use this Python script to train a number plate detector and review the results.

The algorithm of number plate recognition consist as follow : 

- Capturing car images
- Extracting the images of license plate
- Extracting characters from the plate
- Recognizing the characte

We’ll wrap up the post by looking at the results of applying our number plate recognition.


Plate extraction module as shown in Figure 2:

![Coding](https://github.com/nursyafiqahJackson/Artificial-IntelligenceProject/blob/main/download%20(3).jfif)

Figure 2: Phases and individual steps for building a vehicle plate number recognition with computer vision and deep learning 

The plate extractiom process contains five different phases. Each phase performs a segmentation process on the gray image to eliminate the redundant pixels that don't belong to a plate region. 

- Input image : Raw and any formate image will be captured and sent ot the computer. 

- Pre-processing : The image are processed into RGB images using the NTSC method. 

- Plate extraction : Searching for vertical position of the license plate.

- Character segmentation : Identifying the charcter and divided into different images

- Character recognition : The character succesfully occur in the database and can have a desired output. 

Let’s take a look at the dataset we’ll be using to train our COVID-19 face mask detector.


Our plate number recognition dataset as shown in Figure 3:

![Figure 3]( hthttps://github.com/nursyafiqahJackson/Artificial-IntelligenceProject/blob/main/images%20(3).jfif )

Figure 3: A plate number recognition that can detect the plate even from sided trace. 

The dataset we’ll be using here today was created by PyImageSearch reader Adrian Rosebrock.

Our goal is to train a custom deep learning model to detect the vechicle number plate within more than 60 km/h. 

How was our face mask dataset created?
Adrian, like me, was curious how does this machine work and how does it implment. Thus, how does this machine understand which type of traffic crime did human make ? 

To help keep her spirits up, Adrian decided to distract herself by applying computer vision and deep learning to solve a real-world problem:

- Best case scenario — she could use her project to help others
- Worst case scenario — it gave her a much needed mental escape


## D.   PROJECT STRUCTURE

The following directory is our structure of our project:
- $ tree --dirsfirst --
- .
- ├── license_plates
- │   ├── group1
- │   |   ├── 0.01.jpg
- │   |   ├── 0.02.jpg
- │   |   ├── 0.03.jpg
- │   |   ├── 0.04.jpg
- │   |   └── 0.05.jpg
- |   └── group2
- │       ├── 0.01.jpg
- │       ├── 0.02.jpg
- │       └── 0.03.jpg
- ├── pyimagesearch
- │   ├── anpr
- |   |   └── _init_.py
- │   |   └── anpr.py 
- |   └──  _init_.py
- └── ocr_license_plate.py
- 5 directories, 12 files


The dataset/ directory contains the data described.

Three image examples/ are provided so that you can test the static image plate number recognition.

We’ll be reviewing three Python scripts in this tutorial:

- license_plates: Directory containing two sub-directories of JPG images
- anpr.py: Contains the PyImageSearchANPR class responsible for localizing license/number plates and performing OCR
- ocr_license_plate.py: Our main driver Python script, which uses our PyImageSearchANPR class to OCR entire groups of images


## E.  RESULT AND CONCLUSION

The result have been performed to test the machine in various situation. The set of image are taken from :
- complex scenes
- various enviroment
- Different angles. 


![Coding](https://github.com/nursyafiqahJackson/Artificial-IntelligenceProject/blob/main/download%20(4).jfif)
![Coding](https://github.com/nursyafiqahJackson/Artificial-IntelligenceProject/blob/main/images%20(2).jfif)


The performance was succesfull with 91.2% and the characters correctly classified. The purpose system will search the image for the high density edge regions which may contain a license plate. After that, a cleaning and a verification process will be performed on the extracte region to filter out those regions that do no contain a licnese plate. After that, the plate will be passed to the segmentation phase where it will be divided into number of sub-images equal to the number of characters contained in the plate. Finally, each sub-images will be passed to a Multi-Layer Perception Neural Network for identification. 


## G.   PROJECT PRESENTATION 

In this project, you learned how to create a vehicle number plate recognition using OpenCV, Keras/TensorFlow, and Deep Learning.


Our vehicle number plate recognition is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient, making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, Jetosn, Nano, etc.).
(https://www.youtube.com/watch?v=nIq88fVd27k&ab_channel=TanWenXiang)


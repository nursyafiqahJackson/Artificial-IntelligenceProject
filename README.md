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

I’ll then show you how to implement a Python script to train a plate recognition on our dataset using Keras and TensorFlow.

We’ll use this Python script to train a number plate detector and review the results.

The algorithm of number plate recognition consist as follow : 

- Capturing car images
- Extracting the images of license plate
- Extracting characters from the plate
- Recognizing the characte

We’ll wrap up the post by looking at the results of applying our number plate recognition.


Plate extraction module as shown in Figure 2:

![Figure 2](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS0167865519303216&psig=AOvVaw1WoDD7gUYVHV2S4FEz6He9&ust=1618709676149000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCNCQqfSRhPACFQAAAAAdAAAAABAR)
Figure 2: Phases and individual steps for building a vehicle plate number recognition with computer vision and deep learning 

The plate extractiom process contains five different phases. Each phase performs a segmentation process on the gray image to eliminate the redundant pixels that don't belong to a plate region. 

- Input image : Raw and any formate image will be captured and sent ot the computer. 

- Pre-processing : The image are processed into RGB images using the NTSC method. 

- Plate extraction : Searching for vertical position of the license plate.

- Character segmentation : Identifying the charcter and divided into different images

- Character recognition : The character succesfully occur in the database and can have a desired output. 

Let’s take a look at the dataset we’ll be using to train our COVID-19 face mask detector.


Our plate number recognition dataset as shown in Figure 3:

![Figure 3]( https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detection_dataset.jpg )

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
-     └── group2
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



## E   TRAINING THE COVID-19 FACE MASK DETECTION

We are now ready to train system using OPenCv, Phyton and Deep Learning.

From there, open up a terminal, and execute the following command:

- $ python train_mask_detector.py --dataset dataset
- [INFO] loading images...
- [INFO] compiling model...
- [INFO] training head...
- Train for 34 steps, validate on 276 samples
- Epoch 1/20
- 34/34 [==============================] - 30s 885ms/step - loss: 0.6431 - accuracy: 0.6676 - val_loss: 0.3696 - val_accuracy: 0.8242
- Epoch 2/20
- 34/34 [==============================] - 29s 853ms/step - loss: 0.3507 - accuracy: 0.8567 - val_loss: 0.1964 - val_accuracy: 0.9375
- Epoch 3/20
- 34/34 [==============================] - 27s 800ms/step - loss: 0.2792 - accuracy: 0.8820 - val_loss: 0.1383 - val_accuracy: 0.9531
- Epoch 4/20
- 34/34 [==============================] - 28s 814ms/step - loss: 0.2196 - accuracy: 0.9148 - val_loss: 0.1306 - val_accuracy: 0.9492
- Epoch 5/20
- 34/34 [==============================] - 27s 792ms/step - loss: 0.2006 - accuracy: 0.9213 - val_loss: 0.0863 - val_accuracy: 0.9688
- ...
- Epoch 16/20
- 34/34 [==============================] - 27s 801ms/step - loss: 0.0767 - accuracy: 0.9766 - val_loss: 0.0291 - val_accuracy: 0.9922
- Epoch 17/20
- 34/34 [==============================] - 27s 795ms/step - loss: 0.1042 - accuracy: 0.9616 - val_loss: 0.0243 - val_accuracy: 1.0000
- Epoch 18/20
- 34/34 [==============================] - 27s 796ms/step - loss: 0.0804 - accuracy: 0.9672 - val_loss: 0.0244 - val_accuracy: 0.9961
- Epoch 19/20
- 34/34 [==============================] - 27s 793ms/step - loss: 0.0836 - accuracy: 0.9710 - val_loss: 0.0440 - val_accuracy: 0.9883
- Epoch 20/20
- 34/34 [==============================] - 28s 838ms/step - loss: 0.0717 - accuracy: 0.9710 - val_loss: 0.0270 - val_accuracy: 0.9922
- [INFO] evaluating network...

|      |    precision    | recall| f1-score | support |
|------|-----------------|-------|----------|---------|
|with_mask|0.99|1.00|0.99|138|
|without_mask|1.00|0.99|0.99|138|
|accuracy| | |0.99|276|
|macro avg|0.99|0.99|0.99|276|
|weighted avg|0.99|0.99|0.99|276|


![Figure 4](https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detector_plot.png)

Figure 4: Figure 10: COVID-19 face mask detector training accuracy/loss curves demonstrate high accuracy and little signs of overfitting on the data

As you can see, we are obtaining ~99% accuracy on our test set.

Looking at Figure 4, we can see there are little signs of overfitting, with the validation loss lower than the training loss. 

Given these results, we are hopeful that our model will generalize well to images outside our training and testing set.


## F.  RESULT AND CONCLUSION

Detecting COVID-19 face masks with OpenCV in real-time

You can then launch the mask detector in real-time video streams using the following command:
- $ python detect_mask_video.py
- [INFO] loading face detector model...
- INFO] loading face mask detector model...
- [INFO] starting video stream...

[![Figure5](https://img.youtube.com/vi/wYwW7gAYyxw/0.jpg)](https://www.youtube.com/watch?v=wYwW7gAYyxw "Figure5")

Figure 5: Mask detector in real-time video streams

In Figure 5, you can see that our face mask detector is capable of running in real-time (and is correct in its predictions as well.



## G.   PROJECT PRESENTATION 

In this project, you learned how to create a COVID-19 face mask detector using OpenCV, Keras/TensorFlow, and Deep Learning.

To create our face mask detector, we trained a two-class model of people wearing masks and people not wearing masks.

We fine-tuned MobileNetV2 on our mask/no mask dataset and obtained a classifier that is ~99% accurate.

We then took this face mask classifier and applied it to both images and real-time video streams by:

- Detecting faces in images/video
- Extracting each individual face
- Applying our face mask classifier

Our face mask detector is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient, making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, Jetosn, Nano, etc.).

[![demo](https://img.youtube.com/vi/-p7HGwOWxtg/0.jpg)](https://www.youtube.com/watch?v=-p7HGwOWxtg "demo")





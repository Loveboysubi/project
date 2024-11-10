# project
## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.
ii) Perform handwritting detection in an image.
iii) Perform object detection with label in an image.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
## I)Perform ROI from an image
### Step1:
Import necessary packages 
### Step2:
Read the image and convert the image into RGB
### Step3:
Display the image
### Step4:
Set the pixels to display the ROI 
### Step5:
Perform bit wise conjunction of the two arrays  using bitwise_and 
### Step6:
Display the segmented ROI from an image.

## Program:
```
 Developed by : SUBISHESH P
 REG.NO       : 212223230220
```
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and convert the image to RGB format
image = cv2.imread('beach.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Step 3: Create an ROI mask
roi_mask = np.zeros_like(image_rgb)
roi_mask[100:250, 100:250, :] = 255  # Define the ROI region(Dog-face)

# Step 4: Segment the ROI using bitwise AND operation
segmented_roi = cv2.bitwise_and(image_rgb, roi_mask)

# Step 5: Display the segmented ROI
plt.imshow(segmented_roi)
plt.title('Region of Interest')
plt.axis('off')

# Show both images side by side
plt.show()
```

## Output:
![Screenshot 2024-11-10 204146](https://github.com/user-attachments/assets/c3faad4d-3997-4353-bf01-407a16b9b70f)

![Screenshot 2024-11-10 204215](https://github.com/user-attachments/assets/54617f49-4729-4ee4-a76a-55c5fef9d236)

## II)Perform handwritting detection in an image
### Step1:
Import necessary packages 
### Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.
### Step3:
Display the results.

## Program:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
    
image_path = 'handwritten_txt.jpg'
detect_handwriting(image_path)
```
## Output:
![Screenshot 2024-11-10 204230](https://github.com/user-attachments/assets/1549bd2e-8cef-4542-ae32-3871eecbb48d)

## III)Perform object detection with label in an image
### Step1:
Import necessary packages 
### Step2:
Set and add the config_file,weights to ur folder.
### Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)
### Step4:
Create a classLabel and print the same
### Step5:
Display the image using imshow()
### Step6:
Set the model and Threshold to 0.5
### Step7:
Flatten the index,confidence.
### Step8:
Display the result.

## Program:
```
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

model=cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))
img=cv2.imread('objects.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2=127.5
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,0,255),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
```
## Output:
![Screenshot 2024-11-10 203614](https://github.com/user-attachments/assets/c1042577-295b-48d1-bb93-bd43bbd10f33)

## Result:
Thus, a python program using OpenCV for following image manipulations are done successfully.

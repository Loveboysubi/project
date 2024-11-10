
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and convert the image to RGB format
image = cv2.imread('image.jpg')
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

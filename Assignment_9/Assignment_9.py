# python Assignment_9.py

import cv2
import numpy as np

def lantuejoul_skeletonize(image):
    print("start skeleton...")
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    while True:
        eroded = cv2.erode(image, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            break
            
    print("done!!!")
    return skel


# Load the image
image = cv2.imread('input3.jpg', 0) # Read the image in grayscale

# Perform skeletonization
skeleton = lantuejoul_skeletonize(image)

# Display the result
cv2.imshow("Original Image", image)
cv2.imshow("Skeleton", skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()




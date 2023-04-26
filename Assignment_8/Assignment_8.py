# python Assignment_8.py

import cv2
import numpy as np

def otsu_threshold(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute the histogram of the grayscale image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Calculate total number of pixels in the image
    total_pixels = gray.shape[0] * gray.shape[1]

    # Initialize variables
    max_var = 0
    threshold = 0
    sum_pix = 0
    sum_bright = 0

    # Loop through all possible threshold values
    for i in range(256):
        sum_pix += hist[i]
        sum_bright += i * hist[i]

        # Calculate the weight, mean, and variance of the background
        w_b = sum_pix / total_pixels
        mean_b = sum_bright / sum_pix if sum_pix > 0 else 0
        var_b = ((np.arange(i+1) - mean_b) ** 2 * hist[:i+1]).sum() / sum_pix if sum_pix > 0 else 0

        # Calculate the weight, mean, and variance of the foreground
        w_f = (total_pixels - sum_pix) / total_pixels
        mean_f = (sum_bright - sum_pix * mean_b) / (total_pixels - sum_pix) if sum_pix < total_pixels else 0
        var_f = ((np.arange(i+1, 256) - mean_f) ** 2 * hist[i+1:]).sum() / (total_pixels - sum_pix) if sum_pix < total_pixels else 0

        # Calculate the inter-class variance
        inter_var = w_b * w_f * (mean_b - mean_f) ** 2

        # Check if the inter-class variance is greater than the current maximum
        if inter_var > max_var:
            max_var = inter_var
            threshold = i

    # Apply the threshold to the grayscale image and return the result
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh
    
# Load the image
img = cv2.imread('image.jpg')
cv2.imshow('Origional Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Otsu's thresholding method
thresh = otsu_threshold(img)

# Display the thresholded image
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()



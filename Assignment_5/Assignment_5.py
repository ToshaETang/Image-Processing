# python Assignment_5.py

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

# Define image size
size = (500, 500)
# Create a new black image
img = Image.new("RGB", size, "black")
# Define the coordinates for the white square
x0, y0 = 100, 100
x1, y1 = 400, 400
# Draw a white rectangle on the black image
draw = ImageDraw.Draw(img)
draw.rectangle([x0, y0, x1, y1], fill="white")
# Save the image as "white_square.jpg"
img.save("white_square.jpg")

#-----------------------------------------(a) rotation with neighbor interpolation
img = cv2.imread('white_square.jpg')
# Determine the center of the image
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
# Create rotation matrix for 30 degrees rotation
M = cv2.getRotationMatrix2D(center, 30, 1)
# Apply rotation to the image with neighbor interpolation
rotated_img_n = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)
# Show image
cv2.imshow('rotated_img_nearest', rotated_img_n)
cv2.waitKey(0)
# Save the rotated image
cv2.imwrite('rotated_img_nearest.jpg', rotated_img_n)

#-----------------------------------------(b) rotation with bilinear interpolation
# Apply rotation to the image with bilinear interpolation
rotated_img_b = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
# Show image
cv2.imshow('rotated_img_bilinear', rotated_img_b)
cv2.waitKey(0)
# Save the rotated image
cv2.imwrite('rotated_img_bilinear.jpg', rotated_img_b)



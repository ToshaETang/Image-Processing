#python Assignment_1.py

import cv2
import numpy as np

img = cv2.imread('test_input.jpg')
cv2.namedWindow('test_input', cv2.WINDOW_NORMAL)
cv2.imshow('test_input', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("---print test_input---")

zero_channel = np.zeros(img.shape[0:2], dtype = "uint8")
B, G, R = cv2.split(img)

print(B)
print("------------------------------------------")
print(G)
print("------------------------------------------")
print(R)
print("==========================================")

X = (R+G+B)/3

X = X.astype(np.uint8)
print(X)
imgBGR = X

cv2.namedWindow('test_output', cv2.WINDOW_NORMAL)
cv2.imshow("test_output", imgBGR)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("---print test_output---")




# python Assignment_4.py

import cv2
import numpy as np

#讀入並顯示影像 
img = cv2.imread('test_input.jpg')
cv2.imshow('test_input', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
height, width, n = img.shape 
print("影像大小為：",img.shape)

#【Average filter】
img_ave = cv2.blur(img, (3, 3))
#圖片存檔
cv2.imwrite('test_output_ave.jpg', img_ave)

#顯示結果
cv2.imshow('Average filter', img_ave)
cv2.waitKey(0)
cv2.destroyAllWindows()



#【Median filter】
img_med = cv2.medianBlur(img,3)
#圖片存檔
cv2.imwrite('test_output_med.jpg', img_med)

#顯示結果
cv2.imshow('Median filter', img_med)
cv2.waitKey(0)
cv2.destroyAllWindows()


#【Unsharp masking】
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Calculate the unsharp mask
mask = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

# Add the mask to the original image
result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
img_unsharp = cv2.addWeighted(img, 1.5, result, -0.5, 0)

#圖片存檔
cv2.imwrite('test_output_unsharp.jpg', img_unsharp)

cv2.imshow('Unsharp masking', img_unsharp)
cv2.waitKey(0)
cv2.destroyAllWindows()




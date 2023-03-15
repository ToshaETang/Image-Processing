# python Assignment_2_B.py

import math
from math import ceil
import cv2
import numpy as np


#讀入並顯示影像
img = cv2.imread('test_output.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('test_input', cv2.WINDOW_NORMAL)
cv2.imshow('test_input', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#讀取影像大小
height, width = img.shape 
print("影像大小為：",img.shape)


#製作D
d = np.array([[0,56],
							[84,28]])
h=int(ceil(height/2))
w=int(ceil(width/2))					
D = np.tile(d,(h, w))

#Threshold img 
Q=img/3
for i in range(height):
	for j in range(width):
		if ((img[i][j])-85*(Q[i][j])) > D[i][j]:
			img[i][j]=round(Q[i][j])+1
		else:
			img[i][j]=round(Q[i][j])
cv2.imwrite('test_output_B.jpg', img)
print("test_output_B.jpg is SAVED~")


#顯示輸出
cv2.namedWindow('test_output_B', cv2.WINDOW_NORMAL)
cv2.imshow('test_output_B', img)
cv2.waitKey(0)
cv2.destroyAllWindows()






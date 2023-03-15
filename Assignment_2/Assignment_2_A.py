# python Assignment_2_A.py

import math
import cv2
import numpy as np

#讀入並顯示影像
img = cv2.imread('test_input.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('test_input', cv2.WINDOW_NORMAL)
cv2.imshow('test_input', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#讀取影像大小
height, width = img.shape 
print("影像大小為：",img.shape)


#製作D
d = np.array([[0,128,32,160],
							[192,64,224,96],
							[48,176,16,144],
							[240,112,208,80]])
h=int(round(height/4))
w=int(round(width/4))					
D = np.tile(d,(h+1, w+1))


#Threshold img 
for i in range(height):
	for j in range(width):
		if (img[i][j])>D[i][j]:
			img[i][j]=255
		else:
			img[i][j]=0
cv2.imwrite('test_output.jpg', img)
print("test_output.jpg is SAVED~")


#顯示輸出
cv2.namedWindow('test_output', cv2.WINDOW_NORMAL)
cv2.imshow('test_output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()







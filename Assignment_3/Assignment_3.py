# python Assignment_3.py

import cv2
import numpy as np


#讀入並顯示影像 imgC -> imgG
imgC = cv2.imread('test_input.jpg')
imgG = cv2.cvtColor(imgC,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('test_input', cv2.WINDOW_NORMAL)
cv2.imshow('test_input', imgC)
cv2.waitKey(0)
cv2.destroyAllWindows()

zero_channel = np.zeros(imgC.shape[0:2], dtype = "uint8")
B, G, R = cv2.split(imgC)
print(B)
print("--------------------")
print(G)
print("--------------------")
print(R)
print("==========================================")


#製作G’
hist, bins = np.histogram(imgG.flatten(), 256, [0, 256]) # 計算灰度直方圖
cdf = hist.cumsum() # 計算累積分布函數
cdf_scaled = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())) # 將累積分布函數縮放到 0 至 255 之間
imgGG = cv2.LUT(imgG, cdf_scaled) # 將像素值重新映射


#處理影像 (r’,g’,b’) = (r,g,b) X G’ / G
height, width = imgG.shape 
print("影像大小為：",imgC.shape)
print("開始處理影像")

for i in range(height):
	for j in range(width):
		t=imgGG[i][j]/imgG[i][j]
		B[i][j]=(B[i][j])*t
		G[i][j]=(G[i][j])*t
		R[i][j]=(R[i][j])*t


		
print(B)
print("--------------------")
print(G)
print("--------------------")
print(R)


imgC = cv2.merge([B,G,R])

#新圖片存檔
cv2.imwrite('test_output.jpg', imgC)
print("test_output.jpg is SAVED~")


cv2.namedWindow('test_output', cv2.WINDOW_NORMAL)
cv2.imshow('test_output', imgC)
cv2.waitKey(0)
cv2.destroyAllWindows()


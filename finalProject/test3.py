import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
import seaborn as sns; sns.set()
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold

#---------------------------------------------------------------------#

def binarize_image(image_path):
    # 讀取影像
    image = cv2.imread(image_path, 0)  # 以灰階模式讀取影像
    # 使用Otsu方法找到最佳門檻值
    _, binary_image = cv2.threshold(image, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

#---------------------------------------------------------------------#

def process_images_in_folder(folder_path):
    runLengthResult = []

    # 處理資料夾內的影像
    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # [b0,b45,b90,b135,w0,w45,w90,w135]
            RL = [0,0,0,0,0,0,0,0]  

            image_path = os.path.join(folder_path, filename)
            # 進行影像二值化處理
            image = binarize_image(image_path)
            height, width = image.shape
            
            ## 00000
            for x in range(1,height):
                runs=1
                for y in range(1,width):
                    if(image[x][y-1] == image[x][y]):
                        runs += 1
                    else:
                        runs *= runs
                        if(image[x][y]==0):
                            RL[0] = RL[0]+runs
                        else:
                            RL[4] = RL[4]+runs
                        runs = 1
                        
                    if(y==width-1):
                        runs *= runs
                        if(image[x][y]==0):
                            RL[0] = RL[0]+runs
                        else:
                            RL[4] = RL[4]+runs     
            #print("0",RL)
            ## 00000

            ## 454545
            for x in range(1,height):
                runs=1
                steps = x  # max = height-1
                try:
                    for y in range(1,steps):
                        if(image[x-y][y] == image[x-y-1][y+1]):
                            runs += 1
                        else:
                            runs *= runs
                            if(image[x-y][y]==0):
                                RL[1] = RL[1]+runs
                            else:
                                RL[5] = RL[5]+runs
                            runs = 1
                        
                        if(y==steps-1):
                            runs *= runs
                            if(image[x-y][y]==0):
                                RL[1] = RL[5]+runs
                            else:
                                RL[1] = RL[5]+runs
                except:
                    pass
            for y in range(2,width-2):
                runs = 1
                steps = height-y
                try:
                    for x in range(1,steps):
                        if(image[height-x][y+x-1] == image[height-x-1][y+x]):
                            runs += 1
                        else:
                            runs *= runs
                            if(image[height-x-1][y+x]==0):
                                RL[1] = RL[1]+runs
                            else:
                                RL[5] = RL[5]+runs
                            runs = 1

                        if(x==steps-1):
                            runs *= runs
                            if(image[height-x-1][y+x]==0):
                                RL[1] = RL[1]+runs
                            else:
                                RL[5] = RL[5]+runs
                            runs = 1
                except:
                    pass
            #print("45",RL)
            ## 454545

            ## 909090
            for y in range(1,width):
                runs = 1
                for x in range(1,height):
                    if(image[height-x-1][y] == image[height-x-2][y]):
                        runs += 1
                    else:
                        runs *= runs
                        if(image[height-x-2][y]==0):
                            RL[2] = RL[2]+runs
                        else:
                            RL[6] = RL[6]+runs
                        runs = 1

                    if(x==height-2):
                        runs = runs*runs
                        if(image[height-x-2][y]==0):
                            RL[2] = RL[2]+runs
                        else:
                            RL[6] = RL[6]+runs
            #print("90",RL)
            ## 909090

            ## 135135
            for x in range(1,width):
                runs = 1
                steps = x
                w = width
                for y in range(1,steps):
                    w = w-1
                    try:
                        if(image[x-y][w] == image[x-y][w-1]):
                            runs += 1
                        else:
                            runs *= runs
                            if(image[x-y][w]==0):
                                RL[3] = RL[3]+runs
                            else:
                                RL[7] = RL[7]+runs
                            runs = 1
                        if (y==steps-1):
                            runs *= runs
                            if(image[x-y][w]==0):
                                RL[3] = RL[3]+runs
                            else:
                                RL[7] = RL[7]+runs    
                    except:
                        pass
            for y in range(1,width-1):
                runs = 1
                steps = height - y
                w = width - y
                for x in range(1,steps):
                    try:
                        w = w-1
                        if(image[height-x][w] == image[height-x][w-1]):
                            runs += 1
                        else:
                            runs *= runs
                            if(image[height-x][w]==0):
                                RL[3] = RL[3]+runs
                            else:
                                RL[7] = RL[7]+runs
                            runs = 1   
                        if (x == steps-1):
                            runs *= runs
                            if(image[height-x][w]==0):
                                RL[3] = RL[3]+runs
                            else:
                                RL[7] = RL[7]+runs
                    except:
                        pass
            #print("135",RL)
            ## 135135

            runLengthResult.append(RL)

    return runLengthResult

#---------------------------------------------------------------------#

def run_Kmeans(X):

    # k-fold交叉驗證的摺疊數
    k_fold = 30
    kf = KFold(n_splits=k_fold)
    # 儲存每個k值的交叉驗證評估结果
    evaluation_results = {}

    for k in range(10):
        # 儲存當前k值的所有評估指標
        k_evaluation = []
        for train_index, test_index in kf.split(X):
            print("...")
            # 取得訓練集和測試集
            X_train=[]
            X_test=[]
            for i in train_index:
                X_train.append(X[i])
            for i in test_index:
                X_test.append(X[i])
            # 建立並擬合k-means模型
            kmeans = KMeans(n_clusters=10)
            kmeans.fit(X_train)
            # 在測試集上進行預測
            y_pred = kmeans.predict(X_test)
            # 計算評估指標(以輪廓係數為例)
            score = silhouette_score(X_test, y_pred)
            k_evaluation.append(score)
    
        # 計算當前k值的平均評估指標
        average_score = np.mean(k_evaluation)
        # 將结果存到字典
        evaluation_results[k] = average_score
        

    print("  ")
    print("----------------------------")
    for k, score in evaluation_results.items():
        print(f"K={k}, Score={score}")   
    print("----------------------------") 
    print("  ")

'''
def run_Kmeans(X):

            # 建立並擬合k-means模型
    kmeans = KMeans(n_clusters=10)
    labels = kmeans.fit(X)

    print(labels.labels_)

'''


#---------------------------------------------------------------------#  

print("================================")
print("處理開始...")
folder_path_train = r"train"
X = process_images_in_folder(folder_path_train)
run_Kmeans(X)
print("處理結束...")
print("================================")


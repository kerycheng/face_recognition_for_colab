import os
import cv2
import numpy as np
from PIL import Image
from data import facenet

def getImagesWithID(DATA_PATH):

        #-------------載入相關模型--------------#
        dataset = facenet.get_dataset(DATA_PATH)
        
        imagePaths, labels, labels_dict = facenet.get_image_paths_and_labels(dataset) 
        #print(imagePaths) 路徑
        #print(labels) 數量
        #print(labels_dict) ids
        
        faces = []
        Ids = []
        
        for imagePath in imagePaths:
            # 開啟圖像轉換成numpy
            faceImg = Image.open(imagePath).convert('L') # 轉灰階
            # PIL圖像轉換為numpy
            faceNp = np.array(faceImg, 'uint8')
            
            # 從圖片名稱中獲取使用者ID
            ID = int(os.path.split(imagePath)[-1].split('.')[1]) 
            
            # iamge
            faces.append(faceNp)
            # ids
            Ids.append(ID)
            
            #print ID
        return np.array(Ids), np.array(faces) # 回傳ids faces
    
import os
import math
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from data import facenet


def load_MTCNN(DATA_PATH):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            
          #  DATA_PATH = os.path.join(BASE_DIR, "data") #data目錄
            
            IMG_OUT_PATH = os.path.join(DATA_PATH, "dataset") #image目錄
            
            FACENET_DATA_PATH = os.path.join(DATA_PATH, "facenet","20180402-114759","20180402-114759.pb") #dacenet路徑

            datadir = IMG_OUT_PATH # 經過偵測、對齊 & 裁剪後的人臉圖像目錄

            dataset = facenet.get_dataset(datadir) # 取得人臉類別(ImageClass)的列表與圖像路徑

            paths, labels, labels_dict = facenet.get_image_paths_and_labels(dataset) #取得每個人臉的圖像路徑跟ID標籤
            #print (paths) #test
            #print (labels) #test
            #print (labels_dict) #test
            print('Origin: Number of classes: %d' % len(labels_dict)) #人臉種類
            print('Origin: Number of images: %d' % len(paths)) #人臉總數

            #------------載入Facenet模型------------#
            modeldir =  FACENET_DATA_PATH
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            #------------計算人臉特徵向量------------#
            batch_size = 5 # 一次輸入的樣本數量
            image_size = 140  # 要做為Facenet的圖像輸入的大小            
            times_pohto = 10.0  # 每張照片看的次數
            nrof_images = len(paths) # 總共要處理的人臉圖像 
            # 計算總共要跑的批次數
            nrof_batches_per_epoch = int(math.ceil(times_pohto * nrof_images / batch_size))
            # 構建一個變數來保存"人臉特徵向量"
            emb_array = np.zeros((nrof_images, embedding_size)) # <-- Face Embedding

            for i in tqdm(range(nrof_batches_per_epoch)): # 實際訓練 facenet
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
    
            return emb_array, labels, labels_dict
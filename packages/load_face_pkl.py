
import os
import pickle

def load_face_pkl(path, COLAB_PATH):
    
    # 人臉特徵
    with open(os.path.join(COLAB_PATH +'/recognizer','lfw_emb_features.pkl'), 'rb') as emb_features_file:
        emb_features =pickle.load(emb_features_file)
        #print(emb_features)

    # 矩陣
    with open(os.path.join(COLAB_PATH +'/recognizer','lfw_emb_labels.pkl'), 'rb') as emb_lables_file:
        emb_labels =pickle.load(emb_lables_file)
        #print(emb_labels)

    # user_ids
    with open(os.path.join(COLAB_PATH +'/recognizer','lfw_emb_labels_dict.pkl'), 'rb') as emb_lables_dict_file:
        emb_labels_dict =pickle.load(emb_lables_dict_file)
        #print(emb_labels_dict)

        emb_dict = {} # key 是label, value是embedding list
    for feature,label in zip(emb_features, emb_labels):
        # 檢查key有沒有存在
        if label in emb_dict:
            emb_dict[label].append(feature)
        else:
            emb_dict[label] = [feature]
            
    #-------------------測試--------------------#    

    print("人臉特徵數量: {}, shape: {}, type: {}".format(len(emb_features), emb_features.shape, type(emb_features)))
    print("人臉標籤數量: {}, type: {}".format(len(emb_labels), type(emb_labels)))
    print("人臉標籤種類: {}, type: {}", len(emb_labels_dict), type(emb_labels_dict))
    
    return emb_features, emb_labels, emb_labels_dict, emb_dict
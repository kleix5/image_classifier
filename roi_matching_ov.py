import os
import pickle
import time

import cv2
import argparse
import numpy as np
from keras.applications.mobilenet import preprocess_input

from imgOVDoc import model_util_ov
# from model_util import DeepModel


class ImageClassifier:
    def __init__(self):
        self.all_skus = {}
        self.model = model_util_ov.DeepModel()
        self.predict_time = 0
        self.time_search = 0
        self.count_frame = 0
        self.top_k = 5

    def extract_features_from_img(self, cur_img):
        """Судя по названию эта функция извлекает фичи из тестового изображения"""
        cur_img = cv2.resize(cur_img, (224, 224))
        img = preprocess_input(cur_img)
        img = np.expand_dims(img, axis=0)
        feature = self.model.extract_feature(img)
        return feature

    def predict(self, img):
        self.count_frame += 1
        before_time = time.time()
        target_features = self.extract_features_from_img(img)
        self.predict_time += time.time() - before_time
        max_distance = 0
        result_dish = 0

        for dish, features_all in self.all_skus.items():
            for features in features_all:
                cur_distance = self.model.cosine_distance(target_features, features)
                cur_distance = cur_distance[0][0]
                if cur_distance > max_distance:
                    max_distance = cur_distance
                    result_dish = dish

        return result_dish, max_distance

    def add_img(self, img_path, id_img):
        img = cv2.imread(img_path)
        cur_img = img
        feature = self.extract_features_from_img(cur_img)
        if id_img not in self.all_skus:
            self.all_skus[id_img] = []
        self.all_skus[id_img].append(feature)
        return feature

    def remove_by_id(self, id_img):
        if id_img in self.all_skus:
            self.all_skus.pop(id_img)

    def remove_all(self):
        self.all_skus.clear()

    def add_img_from_pickle(self, id_img, pickle_path):
        res = pickle.load(open(pickle_path, 'rb'))
        self.all_skus[id_img] = res

    def get_additional_info(self):
        json_res = {}
        json_res["Extract features, time"] = self.predict_time
        json_res["Find nearest, time"] = self.time_search
        json_res["Count frame"] = self.count_frame
        json_res["RPS"] = self.count_frame / (self.predict_time + self.time_search)
        return json_res


# if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to input image")
    # args = vars(ap.parse_args())

    # img = cv2.imread(args["image"])

def run_match(img_path):
    
    img = cv2.imread(img_path)

    classifaier = ImageClassifier()
    d_img = "uploads/unpack_archive"
    d_path = os.listdir(d_img)

    for f in d_path:
        classifaier.add_img(os.path.join(d_img, f), f)
        
    # t_img = "Otest"
    # t_path = os.listdir(t_img)
    # img = cv2.imread(os.path.join(t_img, t_path[0]))
    result = classifaier.predict(img)
    if result[1] < 0.6:
        return "нет совпадений"
    else:
        return result
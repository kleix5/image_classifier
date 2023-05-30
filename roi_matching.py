import os
import pickle
import time

import cv2
import numpy as np
from keras.applications.mobilenet import preprocess_input

from model_util import DeepModel


class ImageClassifier:
    def __init__(self):
        self.all_skus = {}
        self.model = DeepModel()
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


if __name__ == "__main__":

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frameSize = (int(1920), int(1080))
    new_video = cv2.VideoWriter("test_roi.avi", fourcc=fourcc, fps=25, apiPreference=0,
                                frameSize=frameSize)

    contour_table_test26 = [(1153, 744), (1149, 919), (984, 921), (994, 742)]
    # [{"x": 1153, "y": 744}, {"x": 1149, "y": 919}, {"x": 984, "y": 921}, {"x": 994, "y": 742}]
    contour = np.array(contour_table_test26).reshape((-1, 1, 2)).astype(np.int32)

    x, y, w, h = cv2.boundingRect(contour)

    test_images = 'test'

    classifier = ImageClassifier()

    test_paths = os.listdir(test_images)

    for f in test_paths:
        classifier.add_img(os.path.join(test_images, f), f)

    path_t = 'data/output.mp4'

    cap = cv2.VideoCapture(path_t)
    count_frame = 0
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    start_index = 0
    # bb = ((898, 474), (1326, 700))

    bb = ((x, y), (x + w, y + h))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_cut = frame[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]]
        # frame_cut = frame[y:y + h, x:x + w]
        try:
            name, dist = classifier.predict(frame_cut)
            print(dist)
            print(name)
            if dist > 0.7:
                cv2.rectangle(frame, (int(bb[0][0]), int(bb[0][1])), (int(bb[1][0]), int(bb[1][1])),
                              COLOR_GREEN, 1)
                cv2.putText(frame, str(dist), (int(bb[0][0]), int(bb[1][1]) + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            COLOR_GREEN, 2)
            else:
                cv2.rectangle(frame, (int(bb[0][0]), int(bb[0][1])), (int(bb[1][0]), int(bb[1][1])), COLOR_RED,
                              1)
                cv2.putText(frame, str(dist), (int(bb[0][0]), int(bb[1][1]) + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            COLOR_RED, 2)


        except Exception as e:
            pass

        new_video.write(frame)
        count_frame += 1
        print(count_frame)

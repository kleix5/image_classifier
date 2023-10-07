import os
import numpy as np
import zipfile
import io
import inspect
import cv2
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from openvino.runtime import Core

comparation = {}

core = Core()

classification_model_xml = 'image-retrieval-0001/FP32/image-retrieval-0001.xml'

model = core.read_model(model=classification_model_xml)
compiled_model = core.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def preprocess_image(image_path, h, w):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (w, h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.array(image)
    blob = np.expand_dims(image, axis=0)

    return blob


query_image_path = os.path.join('images', os.listdir('images')[0]) # возвращаем путь к картинке по которой ищем совпадение
preprocessed_query_image = preprocess_image(query_image_path, 224, 224)

result_infer = compiled_model([preprocessed_query_image])[output_layer] # обрабатываем картинку
result_index = np.argmax(result_infer)


# Preprocess the batch of images
batch_images_path = os.path.join('archive', os.listdir('archive')[0])
batch_images_dir = 'unpack_archive'

with zipfile.ZipFile(batch_images_path, 'r') as zip_ref:
    zip_ref.extractall(batch_images_dir)
    zip_ref.close()

batch_images = []
for image_file in os.listdir(batch_images_dir):
    image_path = os.path.join(batch_images_dir, image_file)
    preprocessed_image = preprocess_image(image_path, 224, 224)
    batch_images.append(preprocessed_image)

batch_images = np.array(batch_images)

result_batch = []
for i in range(len(batch_images)):
    result = compiled_model([batch_images[i]])[output_layer]
    result_batch.append(result)
    
result_batch = np.array(result_batch)

for i in range(len(result_batch)):
    similarities = 1 - cosine_distances(result_batch[i], result_infer)
    percentages = similarities * 100
    comparation[os.listdir(batch_images_dir)[i]] = percentages
    
    
best_similarity = max(comparation, key=lambda x: comparation[x])


def answer(similarity):
    result = comparation[similarity].item(0, 0)
    if result > 60:
        print(similarity, result)
        return result
    else:
        return "Нет совпадений!"

answer(best_similarity)
    





























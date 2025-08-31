import pyautogui
import PIL
import time
import os
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn import metrics
import glob

folder_path = "capture_test_run/unsorted"
num_images = 5
for image_num in range(1, num_images + 1):
    tile = pyautogui.screenshot()
    tile.save(os.path.join(folder_path, f"tile{str(image_num)}.png"))
    time.sleep(1)

model = keras.models.load_model('weights/region_classifier_1.h5')

class_names = ['adequate', 'blood', 'clot']

dest_dir_adequate = "capture_test_run/adequate"
dest_dir_blood = "capture_test_run/blood"
dest_dir_clot = "capture_test_run/clot"

img_height = 1000
img_width = 1000

captured_img_list = os.listdir(folder_path)

for cur_img in captured_img_list:
    img_path = os.path.join(folder_path, cur_img)
    img = tf.keras.utils.load_img(img_path, target_size = (img_height, img_width))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    if(class_names[np.argmax(score)] == "adequate"):
        os.rename(img_path, os.path.join(dest_dir_adequate, cur_img)) 
    elif(class_names[np.argmax(score)] == "blood"):
        os.rename(img_path, os.path.join(dest_dir_blood, cur_img))
    elif(class_names[np.argmax(score)] == "clot"):
        os.rename(img_path, os.path.join(dest_dir_clot, cur_img))

yolo_model = YOLO("weights/39_last.pt")

yolo_img_ls = []

if(len(os.listdir("capture_test_run/adequate")) != 0):
    for yolo_norm_file in os.listdir("capture_test_run/adequate"):
        yolo_img_ls.append(os.path.join("capture_test_run/adequate", yolo_norm_file))
    results = yolo_model(yolo_img_ls, save=True)

else:
    print("No adequate regions!")

files = glob.glob("capture_test_run/*/*")

for f in files:
    os.remove(f)
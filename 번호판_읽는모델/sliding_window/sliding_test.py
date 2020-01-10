from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import csv, operator
from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from keras.models import model_from_json
# 0. 사용할 패키지 불러오기
import numpy as np
import glob, os
from PIL import Image
from keras.models import Sequential, load_model
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy
import pandas as pd

csvfile = open('responsemap.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(csvfile)

categories = ["0", '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
              'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
              'T', 'U', 'V', 'W', 'X', 'Y']
caltech_dir = "/home/pirl/LP_detection_model/license_plate/sliding_window/images"
image_w = 12
image_h = 20
pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)
X = X.astype(float)/255

json_file = open("/home/pirl/opencv/SangWoo_Lab/license_plate/results/model1205.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/home/pirl/opencv/SangWoo_Lab/license_plate/results/w1205.h5")
print("Loaded model from disk")


lr = 0.001
# optimizer Adam
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999)

loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

prediction = loaded_model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.9f}".format(x)})
cnt = 0
answer = ''
for i in prediction:
    if all(i < 0.89):
        answer = '*'
    else:
        pre_ans = i.argmax()  # 예측 레이블
        pre_ans_str = ''
        answer = categories[pre_ans]

    print(filenames[cnt],"의 예측값은 ",answer)
    print(i)




    # csv 파일 생성
    writer.writerow([filenames[cnt].split('/')[-1],answer])
    cnt += 1

csvfile.close()
reader = csv.reader(open("responsemap.csv"), delimiter=",")
sortedlist = sorted(reader, key=operator.itemgetter(0))

char_list = []
temp_list = []
final_list = []

with open("responsemap.csv","w") as f:
    fileWriter = csv.writer(f, delimiter=",")
    for row in sortedlist:
        fileWriter.writerow(row)
        # print(row[1])

        # 예측된 값을 배열로 넣는다
        char_list.append(row[1])

#print(char_list)

for i in char_list:
    # print(i)
    if i != '*':
        temp_list.append(i)

# print(temp_list)

for i in temp_list:
    try:
        if temp_list[i] == temp_list[i+1]:
            final_list.append(temp_list[i])
    except:print("error")

print(final_list)


def predictImage(imgroute, model=loaded_model, categories=categories, image_h = 20, image_w=12):
    img = Image.open(imgroute)
    img = img.convert("RGB")
    img = np.asarray(img)
    img = img.astype(float)/255
    prediction = model.predict(img)
    p_class = prediction.argmax()
    p_class = categories[p_class]
    print("==p_class",p_class)
    return p_class

# predictImage("/home/pirl/PycharmProjects/keraslp/sliding-window/images/0_153.jpg")
import glob
import cv2
import time
import csv

# y = H = 높이 = image.shape[0] = 행렬의 높이
# x = W = 가로 = image.shape[1] = 행렬의 가로


def sliding(image, i, step_y, step_x, windowsize):
    cnt = 0
    print("전체이미지 크기 : ", image.shape)
    print("window size : ", windowsize[0], windowsize[1])
    for y in range(0, image.shape[0], step_y):
        for x in range(0, image.shape[1], step_x):
            if y+ windowsize[0] > image.shape[0] or x + windowsize[1] > image.shape[1]:
                continue
            else:
                cnt += 1
                if cnt < 10:
                    number = str('00')+str(cnt)
                elif cnt < 100:
                    number = str('0')+str(cnt)
                else:
                    number = cnt
                img_name = str(i) + '_{}.jpg'.format(number)
                img_dir = './images/'+img_name
                cv2.imwrite(img_dir, image[y:y+windowsize[0], x:x+windowsize[1]])
                print(i,"번째 번호판의 ", "image{} 생성".format(number), "좌표 : ",(y,y+windowsize[0],x,x+windowsize[1]))
                cv2.waitKey(1)
                time.sleep(0.025)


image_dir = "./sliding_window_test"
files = glob.glob(image_dir + "/*.jpg")
for i, f in enumerate(files):
    image = cv2.imread(f)
    sliding(image,i, step_y = 2, step_x = 2, windowsize = (20,12))
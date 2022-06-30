import cv2
import numpy as np
from copy import deepcopy
import os

h_list = [390,300,240,300,300,300]
hg_list = [30,30,20,10,10,0]
w_list = [135.25,135.25,135.25,135.25,135.25,135.25]
wg_list = [20,10,10,10,20,0]
file_path = "C:\\Users\\Hyeonseok\\Desktop\\KIROdata\\2set_Video_Left\\"
# data_path = "04-pressure_mat/KIRO_0.jpg"
#
images=[]
for image in os.listdir(file_path):
    images.append(image)
os.chdir(file_path)
for idx in range (0,len(images)):
    print(images[idx])

    img = cv2.imread(images[idx])
    img = np.array(img)
    img = img * 150
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,dsize=(882,1930))


    img_data = deepcopy(np.array(img))

    zero_pad = np.zeros([6,6])

    i_idx = 0
    print(np.shape(zero_pad),np.max(zero_pad))
    for i in range(0,6):
        j_idx =0

        for j in range(0, 6):
            tmp_img = img_data[i_idx:i_idx+ h_list[i],int(j_idx):int(j_idx+w_list[j])]
            j_idx += wg_list[j] + w_list[j]
            zero_pad[i,j] = np.average(tmp_img)
            #print(np.average(tmp_img))
        i_idx += hg_list[i] + h_list[i]

    zero_pad = zero_pad.astype(np.uint8)
    result = cv2.resize(zero_pad,dsize=(32,64))
    # result = cv2.resize(zero_pad,dsize=(400,900))


    org_pad = np.zeros([1930,882])
    i_idx= 0
    for i in range(0,6):
        j_idx =0
        for j in range(0, 6):
            #org_pad[i_idx:i_idx+ h_list[i], int(j_idx):int(j_idx+w_list[j])] =1
            value = zero_pad[i,j]

            one_pad = np.ones([h_list[i], int(j_idx + w_list[j]) - int(j_idx)])
            one_pad *= zero_pad[i,j]
            org_pad[i_idx:i_idx + h_list[i], int(j_idx):int(j_idx + w_list[j])] = one_pad

            j_idx += wg_list[j] + w_list[j]
            #print(np.average(tmp_img))

        i_idx += hg_list[i] + h_list[i]
    org_pad = cv2.resize(org_pad,dsize=(32,64))
    # org_pad = cv2.resize(org_pad,dsize=(400,900))
    org_pad = np.array(org_pad,dtype=np.uint8)
    show_img = cv2.resize(img,dsize=(32,64))
    # show_img = cv2.resize(img,dsize=(400,900))

    cv2.imshow("show_img",show_img)
    cv2.imshow("org_pad",org_pad)
    cv2.imshow("result",result)
    cv2.waitKey()
    # cv2.imwrite("C:\\Users\\Hyeonseok\\Desktop\\KIROdata\\2set_Low_image_Left_32_64\\"+images[idx], result)
    # cv2.imwrite("D:\AItest/result/"+images[idx],result)
    # cv2.imwrite("D:\AItest/org_big/"+images[idx],show_img)
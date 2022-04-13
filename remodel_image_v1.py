import cv2
import util
import numpy as np

raw_path = "DATA\\image\\data3\\orignal\\"
target_path = "DATA\\image\\data3\\target\\"
train_output_path2 = "DATA\\image\\data3\\label1\\"

train_input_path = "DATA\\train-data_test\\input\\"
train_output_path = "DATA\\train-data_test\\output\\"
kernel = np.ones((7,7), np.uint8)

def check_folder():
    # if Dir is not exist, generate dir.
    if not (util.check_folder(train_input_path)):
        util.make_dir_folder(train_input_path)
    if not (util.check_folder(train_output_path)):
        util.make_dir_folder(train_output_path)
    if not (util.check_folder(train_output_path2)):
        util.make_dir_folder(train_output_path2)

def run():
    raw_list = util.check_dirction_path(raw_path)   ## 이 path 의 data list 를 갖고옴

    for data in raw_list:                           ## raw_list 순서대로 data 동작함
        save_name = str(data)
        print(save_name)

        # TODO: try exception 으로 동일한 파일이없을 때 찾아야함
        tmp_input = cv2.imread(raw_path + data)
        tmp_output = cv2.imread(target_path + data)
        save_input = tmp_input

        height = tmp_input.shape[0]
        width = tmp_input.shape[1]

        # height = tmp_output.shape[0]
        # width = tmp_output.shape[1]
        save_output = np.zeros([height,width])

        # target 데이터 받아서 3개의 gray scale 로 변환

        for i in range(0, height):
            for j in range(0, width):
                if (tmp_output[i, j, 0] > 0 and tmp_output[i, j, 1] < 200 and tmp_output[i, j, 2] < 200):
                    save_output[i, j] = 50
                elif (tmp_output[i, j, 0] < 200 and tmp_output[i, j, 1] > 0 and tmp_output[i, j, 2] < 200):
                    save_output[i, j] = 100
                elif (tmp_output[i, j, 0] < 200 and tmp_output[i, j, 1] < 200 and tmp_output[i, j, 2] > 0):
                    save_output[i, j] = 150

                elif (tmp_output[i, j, 0] > 0 and tmp_output[i, j, 1] > 0 and tmp_output[i, j, 2] < 200):
                    if (tmp_output[i, j, 0] > tmp_output[i, j, 1]):
                        save_output[i, j] = 0
                    else:
                        save_output[i, j] = 50

                elif (tmp_output[i, j, 0] < 200 and tmp_output[i, j, 1] > 0 and tmp_output[i, j, 2] > 0):
                    if (tmp_output[i, j, 1] > tmp_output[i, j, 2]):
                        save_output[i, j] = 0
                    else:
                        save_output[i, j] = 100

                elif (tmp_output[i, j, 0] > 0 and tmp_output[i, j, 1] < 200 and tmp_output[i, j, 2] > 0):
                    if (tmp_output[i, j, 0] > tmp_output[i, j, 2]):
                        save_output[i, j] = 0
                    else:
                        save_output[i, j] = 150
                else:
                    save_output[i, j] = 0;

        save_output = cv2.morphologyEx(save_output, cv2.MORPH_CLOSE, kernel)

        #cv2.imwrite(train_input_path + save_name, save_input)
        # cv2.imwrite(train_output_path + save_name, save_output)
        cv2.imwrite(train_output_path2 + save_name, save_output)

if __name__ == "__main__":
    check_folder()
    run()
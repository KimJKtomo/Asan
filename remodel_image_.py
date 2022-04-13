import cv2
import util
import numpy as np

raw_path = "C:\\Users\\Hyeonseok\\Desktop\\DATA\\data2\\"
train_input_path = "C:\\Users\\Hyeonseok\\Desktop\\DATA\\train-data2_1\\input\\"
train_output_path = "C:\\Users\\Hyeonseok\\Desktop\\DATA\\train-data2_1\\output\\"
test_input_path = "C:\\Users\\Hyeonseok\\Desktop\\DATA\\train-data2_1\\input\\"
test_output_path = "C:\\Users\\Hyeonseok\\Desktop\\DATA\\train-data2_1\\output\\"
kernel = np.ones((7,7), np.uint8)
def check_folder():
    if not (util.check_folder(train_input_path)):
        util.make_dir_folder(train_input_path)
    if not (util.check_folder(train_output_path)):
        util.make_dir_folder(train_output_path)

def run():
    raw_list = util.check_dirction_path(raw_path)
    raw_input = []
    raw_output = []
    for data in raw_list:

        if data.split("_")[-1]  == "L.jpg":
            #tmp_data = data.split(".")[0][:-2] + "." + data.split(".")[1]
            raw_output.append(data)
        else:
            raw_input.append(data)

    for idx in range(0,len(raw_input)):

        save_name = "frame" + str((idx)) + ".png"
        print(save_name)
        tmp_in_data = raw_input[idx]
        print(tmp_in_data)
        tmp_out_data = raw_output[idx]
        tmp_input = cv2.imread(raw_path + tmp_in_data)
        tmp_output = cv2.imread(raw_path + tmp_out_data)
        height = tmp_input.shape[0]
        width = tmp_input.shape[1]

        # save_input = tmp_input[:,160:width-160,:]
        # tmp_output = tmp_output[:, 160:width - 160, :]
        save_input = tmp_input
        tmp_output = tmp_output

        height = tmp_output.shape[0]
        width = tmp_output.shape[1]

        save_output = np.zeros([height,width])
        backround = np.ones([height, width])
        """
        for i in range(0,3):
            tmp_output[:,:,i] = background*255 - tmp_output[:,:,i]
        for i in range(0, 3):
            save_output += tmp_output[:,:,2-i]/255 * 30*i
        save_output = save_output.astype(np.int8)
        """
        for i in range(0,height):
            for j in range(0,width):
                if(tmp_output[i,j,0]>0 and tmp_output[i,j,1]<200 and tmp_output[i,j,2]<200):
                    save_output[i,j] = 50
                elif (tmp_output[i, j, 0] < 200 and tmp_output[i, j, 1] > 0 and tmp_output[i, j, 2] < 200):
                    save_output[i, j] = 100
                elif (tmp_output[i, j, 0] < 200 and tmp_output[i, j, 1] < 200 and tmp_output[i, j, 2] > 0):
                    save_output[i, j] = 150

                elif (tmp_output[i, j, 0] > 0 and tmp_output[i, j, 1] > 0 and tmp_output[i, j, 2] < 200):
                    if ( tmp_output[i, j, 0] >tmp_output[i, j, 1]): save_output[i, j] = 0
                    else: save_output[i, j] = 50

                elif (tmp_output[i, j, 0] < 200 and tmp_output[i, j, 1] > 0 and tmp_output[i, j, 2] > 0):
                    if ( tmp_output[i, j, 1] >tmp_output[i, j, 2]): save_output[i, j] = 0
                    else: save_output[i, j] = 100

                elif (tmp_output[i, j, 0] > 0 and tmp_output[i, j, 1] < 200 and tmp_output[i, j, 2] > 0):
                    if ( tmp_output[i, j, 0] >tmp_output[i, j, 2]): save_output[i, j] = 0
                    else: save_output[i, j] = 150

                else:
                    save_output[i, j] = 0;

        save_output = cv2.morphologyEx(save_output, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(train_input_path + save_name, save_input)
        cv2.imwrite(train_output_path + save_name, save_output)


if __name__ == "__main__":
    check_folder()
    run()
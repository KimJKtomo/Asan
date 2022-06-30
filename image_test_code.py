from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import time
import argparse

categories = ["Supine", "Right", "Left","Others"]

def test_model(input_image,input_label, model):
    # caltech_dir = "./test_image"
    caltech_dir =input_image
    label_dir = input_label
    image_w = 32
    image_h = 64

    pixels = image_h * image_w * 3
    image = []
    X = []
    Y = []
    filenames = []

    files = glob.glob(caltech_dir+"/*.*")
    for i, f in enumerate(files):
        img = Image.open(f)
        angle = 360
        img = img.rotate(angle, expand=True)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        image.append(img)
        filenames.append(f)
        X.append(data)

    labels = glob.glob(label_dir+"/*.*")
    for i_2, f_2 in enumerate(labels):
        label_check =  open(f_2)
        line = label_check.readline().splitlines()
        Y.append(line)
    # print(Y)
    b = []
    for i_3 in range(len(Y)):
        for j in range(len(Y[0])):
            b.append(int(Y[i_3][j]))
    print(b)

    X = np.array(X)
    # model = load_model('D:\\KJG_AItest\\model\\For_test.model')
    model = load_model(model)
    prediction = model.predict(X)
    print(prediction)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    cnt = 0
    Yes = 0
    No = 0
    Supine = 0
    Right = 0
    Left = 0
    Others = 0
    print("prediction = ", prediction)

    # np.savetxt("C:\\Users\\Hyeonseok\\Desktop\KIROdata\\2set\\Left\\Posture_Video_left\\Posture_Video_left.txt", prediction, fmt='%2d', delimiter=',')

    #이 비교는 그냥 파일들이 있으면 해당 파일과 비교. 카테고리와 함께 비교해서 진행하는 것은 _4 파일.
    for val in prediction:
        pre_ans = val.argmax()  # 예측 레이블
        print(val)
        print(pre_ans)
        pre_ans_str = ''
        Orignal=''
        print(image[cnt])
        plt.imshow(image[cnt])
        print("b[cnt] =", b[cnt])
        if b[cnt] == 0:
            Orignal = "Supine"
        elif b[cnt] == 1:
            Orignal = "Right"
        elif b[cnt] == 2:
            Orignal = "Left"
        else: Orignal = "Others"

        if pre_ans == 0:
            Supine += 1
        elif pre_ans == 1:
            Right += 1
        elif pre_ans == 2:
            Left += 1
        else:
            Others += 1

        if pre_ans == 0: pre_ans_str = "Supine"
        elif pre_ans == 2: pre_ans_str = "Left"
        elif pre_ans == 1: pre_ans_str = "Right"
        else: pre_ans_str = "Others"

        if val[0] >= 0.8 : print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
        if val[1] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"으로 추정됩니다.")
        if val[2] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
        if val[3] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")

        cnt += 1
        # plt.xticks([])
        # plt.yticks([])
        print(cnt)
        plt.xlabel("AI Expected label: "+pre_ans_str+"\n Frames "+str(cnt))
        plt.xlabel("Original label: "+Orignal+"\n AI Expected label: "+pre_ans_str+".")
        plt.tight_layout()

        # plt.savefig('C:\\Users\\Hyeonseok\\Desktop\\KIROdata\\2set\\Left\\Posture_Video_left\\Posture_Video_left%d.png' % cnt , bbox_inches='tight')
        if divmod(cnt+1,5)[1] == 0:
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        else:
            plt.close()
        # plt.show()

        print(Orignal)

        if Orignal == pre_ans_str:
            Yes += 1
        else:
            No += 1


    Result = (100 * Yes / (Yes + No))
    Bar = [Yes, No]
    index = np.arange(len(Bar))


    plt.bar(index, Bar)
    plt.xticks([0,1],['Yes', 'No'])
    plt.title('test Result')
    plt.xlabel(" Result = " + str(Result) + "%")
    # plt.savefig('./Result_image/Posture Result Accuracy.png')
    plt.show()
    plt.pause(1)
    plt.close()

    print(" Result Accuracy = " + str(Result) + "%")

        # cv2.destroyAllWindows()

if __name__ == "__main__":
    start = time.time()  # 시작 시간 저장

    parser = argparse.ArgumentParser()

    parser.add_argument('-img', '--input_image', action='store', dest='input_image', default='C:/Users/Hyeonseok/Desktop/dcscn-super-resolution-ver1/SRGAN/Keras-SRGAN/test_image_20211222',
                        help='Path for input images')
    parser.add_argument('-L', '--input_label', action='store', dest='input_label', default='C:/Users/Hyeonseok/Desktop/dcscn-super-resolution-ver1/SRGAN/Keras-SRGAN/test_label',
                        help='Path for input labels')
    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir',
                        default='C:/Users/Hyeonseok/Desktop/dcscn-super-resolution-ver1/SRGAN/Keras-SRGAN/model_posture/Opendata_img_classification_All_tilt.model',
                        help='Path for model')


    values = parser.parse_args()


    model = load_model(values.model_dir)
    print(model.summary())
    model_end = time.time()  # 시작 시간 저장


    test_model(values.input_image, values.input_label,  values.model_dir)

    print("Model time :", model_end - start)  # 현재시각 - 시작시간 = 실행 시간
    print("Super Resoluton time :", time.time() - model_end)  # 현재시각 - 시작시간 = 실행 시간
    print("total time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    # history = model.fit(X_train, y_train, batch_size=8, epochs=20, validation_split=0.2,
    #                     callbacks=[checkpoint, early_stopping])
    # print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))

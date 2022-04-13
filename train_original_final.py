import math
import random
import os
import time
from math import ceil
from copy import deepcopy
import cv2
import matplotlib
import util

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from imgaug import augmenters as iaa
from imgaug import imgaug
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import io

train_input_path = "C:\\Users\\Hyeonseok\\Desktop\\DATA\\train-data2_1\\"

class Network:
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    IMAGE_CHANNELS = 4
    IMAGE_INDEX = 4

    def __init__(self, layers=None, per_image_standardization=True, batch_norm=True, skip_connections=True):
        # Define network - ENCODER (decoder will be symmetric).
        with tf.device('gpu:0'):
            if layers == None:
                layers = []
                layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
                layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
                layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=skip_connections))

                layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_2_1'))
                layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2'))
                layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=skip_connections))

                layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_3_1'))
                layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2'))
                layers.append(MaxPool2d(kernel_size=2, name='max_3'))

            self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                         name='inputs')
            self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_INDEX],
                                          name='targets')
            self.is_training = tf.placeholder_with_default(False, [], name='is_training')
            self.description = ""

            self.layers = {}

            if per_image_standardization:
                list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
                net = tf.stack(list_of_images_norm)
            else:
                net = self.inputs

            # ENCODER
            for layer in layers:
                self.layers[layer.name] = net = layer.create_layer(net)
                self.description += "{}".format(layer.get_description())

            print("Current input shape: ", net.get_shape())

            layers.reverse()
            Conv2d.reverse_global_variables()

            # DECODER
            for layer in layers:
                net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

            self.segmentation_result = tf.sigmoid(net)

            # segmentation_as_classes = tf.reshape(self.y, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH, 1])
            # targets_as_classes = tf.reshape(self.targets, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
            # print(self.y.get_shape())
            # self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(segmentation_as_classes, targets_as_classes))
            print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                            self.targets.get_shape()))

            # MSE loss
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))
            self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)

            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()


class Dataset:
    def __init__(self, batch_size=10, folder=train_input_path, size=[512, 512], index=4):
        self.batch_size = batch_size
        self.size = size
        self.folder = folder
        self.index = index
        # self.crop_pos = [(0, 0), (0, 448), (208, 0), (208, 448)]
        # self.crop_pos = []
        train_files, validation_files, test_files = self.train_valid_test_split(
            os.listdir(os.path.join(folder, 'input')))

        self.train_inputs, self.train_targets = self.file_paths_to_images(train_files)
        self.test_inputs, self.test_targets = self.file_paths_to_images(test_files)

        self.pointer = 0

    def file_paths_to_images(self, files_list, verbose=True):
        inputs = []
        targets = []

        for file in files_list:
            input_path = os.path.join(self.folder, 'input', file)
            target_path = os.path.join(self.folder, 'output', file)
            input_image = np.array(cv2.imread(input_path))  # load input data
            output_image = cv2.imread(target_path)  # load target data with gray_scale
            [row, column, w] = input_image.shape
            # a = np.random.randint(row - 512)
            b = np.random.randint(column - 512)
            New_input_image = input_image[:, b:b + 512, :]
            New_input_image = np.array(New_input_image, dtype=np.uint8)
            New_target_image = output_image[:, b:b + 512, :]
            New_target_image = np.array(New_target_image, dtype=np.uint8)

            inputs.append(New_input_image)
            targets.append(New_target_image)

        return inputs, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.7, 0.2, 0.3)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def create_batch(self):
        self.batch_input = []
        self.batch_target = []
        for _ in range(0, 1):
            for idx in range(0, len(self.train_inputs)):
                input_image = self.train_inputs[idx]
                target_image = self.train_targets[idx]
                input_image, target_image = self.shuffle(input_image, target_image)
                input_image, target_image = self.wrapper(input_image, target_image)
                tmp_image = np.array(input_image[:, :, :3], np.uint8)
                self.batch_input.append(input_image)
                self.batch_target.append(target_image)

    def next_batch(self):
        inputs = []
        targets = []
        # print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        for i in range(self.batch_size):
            input_image = self.batch_input[self.pointer + i]
            target_image = self.batch_target[self.pointer + i]
            inputs.append(input_image)
            targets.append(target_image)

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    def shuffle(self, input, target):
        op_1 = np.random.randint(4)    # 0 : 아무것도 안함(crop) , 1 : Geometry (resize, rotation, traslation) , 2 : Noise ( Gaussaion, burring, brightness, Salt&Pepper) ,  3 : ALL
        op_1 = 1;
        if op_1 == 0:
            input, target = self.crop(input, target, action="crop")
        if op_1== 1:
            input, target = self.Geometry(input, target)
        if op_1 == 2:
            input, target = self.NoiseAction(input, target)
        if op_1 == 3:
            input, target = self.Geometry(input, target)
            input, target = self.NoiseAction(input, target)

        return input, target

    def Geometry(self, input, target):
        op_2 = np.random.randint(3);
        op_2 = 0;
        if op_2 == 0:
            input, target = self.crop(input, target, action="resize", resize=[1, 1.07])
        if op_2 == 1:
            input, target = self.modify(input, target, action="rotation")
            [width, height] = [512, 512]
            M = cv2.getRotationMatrix2D((width / 2, height / 2), angle=np.random.randint(360), scale=1)
            input = cv2.warpAffine(input, M, (width, height))
            target = cv2.warpAffine(target, M, (width, height))
        if op_2 == 2:
            input, target = self.translation(input, target)
        return input, target

    def translation(self, input, target):
        M = np.float32([[1, 0, np.random.randint(-100, 100)], [0, 1, np.random.randint(-100, 100)]])
        rows, cols= np.shape(input)[:2]
        dst1 = cv2.warpAffine(input, M, (cols, rows))
        rows2, cols2 = np.shape(target)[:2]
        dst2 = cv2.warpAffine(target, M, (cols2, rows2))
        return dst1, dst2

    def NoiseAction(self, input, target):
        op_3 = np.random.randint(4)
        if op_3 == 0:
            input, target = self.GaussaionNoise(input, target)
        if op_3 == 1:
            input, target = self.sp_noise(input, target)
        if op_3 == 2:
            input, target = self.brightness(input, target)
        if op_3 == 3:
            input, target = self.burring(input, target)
        return input, target

    def GaussaionNoise(self, input, target):
        row, col, ch = input.shape
        row2, col2 ,ch2= target.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = input + gauss
        gauss = np.random.normal(mean, sigma, (row2, col2, ch2))
        gauss = gauss.reshape(row2, col2, ch2)
        noisy_2 = target + gauss
        return noisy, noisy_2

    def sp_noise(self, image, target):
        output_1 = np.zeros(image.shape, np.uint8)
        output_2 = np.zeros(target.shape, np.uint8)
        thres = 1 - 0.05
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < 0.05:
                    output_1[i][j] = 0
                elif rdn > thres:
                    output_1[i][j] = 255
                else:
                    output_1[i][j] = image[i][j]
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                rdn = random.random()
                if rdn < 0.05:
                    output_2[i][j] = 0
                elif rdn > thres:
                    output_2[i][j] = 255
                else:
                    output_2[i][j] = target[i][j]
        return output_1, output_2

    def brightness(self, input, target):
        new_input = np.zeros(input.shape, input.dtype)
        new_target = np.zeros(target.shape, target.dtype)
        beta = 30
        for y in range(input.shape[0]):
            for x in range(input.shape[1]):
                for c in range(input.shape[2]):
                    new_input[y, x, c] = np.clip(input[y, x, c] + beta, 0, 255)
        input=new_input
        for y in range(target.shape[0]):
            for x in range(target.shape[1]):
                new_target[y, x] = np.clip(target[y, x] + beta, 0, 255)
        target = new_target

        return input, target

    def burring(self, input, target):
        input_blur = cv2.blur(input,(25,25))
        target_blur = cv2.blur(target,(25,25))

        return input_blur, target_blur

    def augmentation(self, batch_inputs, value=0.1):
        augmentation_seq = iaa.Sequential([
            iaa.GaussianBlur((0, 3.0 * value), name="GaussianBlur"),
            iaa.Dropout(0.1 * value, name="Dropout"),
            iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="GaussianNoise")])

        augmentation_seq_deterministic = augmentation_seq.to_deterministic()
        batch_inputs = augmentation_seq_deterministic.augment_images(batch_inputs)
        return batch_inputs

    def crop(self, input, target, action="crop", pos=[0, 0]):
        if action == "crop":
            input = input[pos[0]:pos[0] + 512, pos[1]:pos[1] + 512]
            target = target[pos[0]:pos[0] + 512, pos[1]:pos[1] + 512]

        return input, target

    def crop(self, input, target, action="resize", resize=[1.06, 1]):
        if action == "resize":
            input = cv2.resize(input, (0, 0), fx=resize[0], fy=resize[1])
            target = cv2.resize(target, (0, 0), fx=resize[0], fy=resize[1])
            input = input[:self.size[0], :self.size[1], :]
            target = target[:self.size[0], :self.size[1],:]
        return input, target

    def modify(self, input, target, action="", angle=90, flip=0):
        if action == "rotation":
            [width, height, r] = input.shape
            M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            input = cv2.warpAffine(input, M, (0, 0))
            target = cv2.warpAffine(target, M, (0, 0))
        if action == "flip":
            input = cv2.flip(input, flip)
            target = cv2.flip(target, flip)

        return input, target
    #
    # def wrapper(self, input_image, target_image):
    #     img_size = deepcopy(self.size)
    #     img_size.append(self.index)
    #     result_input = np.zeros(img_size)
    #     result_target = np.zeros(img_size)
    #
    #     [high, width, w] = input_image.shape
    #     result_input[:high, :width, :w] = np.array(input_image, dtype=np.uint8)
    #
    #     [high, width, w] = target_image.shape
    #     for idx in range(0, 4):
    #         result_target[:high, :width, idx] += target_image[:, :,idx] == idx
    #
    #     return np.array(result_input, dtype=np.uint8), np.array(result_target, dtype=np.uint8)

    def wrapper(self, input_image, target_image):
        img_size = deepcopy(self.size)
        img_size.append(self.index)
        result_input = np.zeros(img_size)
        result_target = np.zeros(img_size)

        [high, width, w] = input_image.shape
        result_input[:high, :width, :w] = np.array(input_image, dtype=np.uint8)

        [high, width, w] = target_image.shape
        target_image = np.divide(target_image, 50)
        target_image = np.array(target_image, dtype=np.uint8)
        for idx in range(0, w):
            result_target[:high, :width, idx] += target_image[:, :, idx] == idx

        return np.array(result_input, dtype=np.uint8), np.array(result_target, dtype=np.uint8)

    def test_set(self, size=20):
        inputs = []
        targets = []
        if size > len(self.test_inputs):
            size = len(self.test_inputs)

        for idx in range(0, size):
            input = self.test_inputs[idx]
            target = self.test_targets[idx]

            input , target = self.shuffle(input, target)
            input, target = self.wrapper(input, target)
            inputs.append(input)
            targets.append(target)

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)


def save_test_image(path, inputs, results):
    if len(inputs) == len(results):
        for idx in range(0, len(inputs)):
            save_name = "result_" + str(idx) + ".png"
            ori_name = "orignal_" + str(idx) + ".png"
            tmp_input = inputs[idx][:, :, :3]
            oringal_img = inputs[idx][:, :, :3]
            tmp_results = np.argmax(results[idx], 2)
            tmp_target = np.zeros(tmp_input.shape)
            for i in range(1, 4):
                tmp_target[:, :, i - 1 ] += tmp_results == i

            # tmp_target =results[idx][:,:,1:4]
            tmp_input = np.multiply(tmp_input, 0.5)
            tmp_input = np.array(tmp_input, dtype=np.int8)
            tmp_target = np.array(tmp_target, dtype=np.int8)

            tmp_target = np.multiply(tmp_target, 100, dtype=np.int8)

            tmp_target = np.array(tmp_target, dtype=np.uint8)
            oringal_img = np.array(oringal_img, dtype=np.uint8)

            save_result = cv2.addWeighted(oringal_img, 1, tmp_target, 1, 0)
            cv2.imwrite(path + target_name, tmp_target)
            cv2.imwrite(path + save_name, save_result)
            cv2.imwrite(path + ori_name, oringal_img)


def train():
    network = Network()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    print(timestamp)
    dataset = Dataset(batch_size=10, size=[network.IMAGE_HEIGHT, network.IMAGE_WIDTH])

    # inputs, targets = dataset.next_batch()

    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = False

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
                                               graph=tf.get_default_graph())
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
        test_accuracies = []
        # Fit all training data
        n_epochs = 100
        global_start = time.time()
        for epoch_i in range(n_epochs):
            dataset.pointer = 0
            if epoch_i % 10 == 0:
                dataset.reset_batch_pointer()
                dataset.create_batch()
            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
                # for idx in range(0,10):
                #
                #     target_image = np.multiply(batch_targets[idx],255,dtype=np.uint8)
                #     orignal_image = np.array(batch_inputs[idx],dtype=np.uint8)
                # cv2.imshow("orignal", orignal_image)
                # cv2.imshow("target", target_image)
                # cv2.waitKey()
                # continue
                start = time.time()
                batch_inputs, batch_targets = dataset.next_batch()
                # print(np.max(batch_inputs))
                # print(np.max(batch_targets))
                # cv2.imshow("batch_inputs", batch_inputs[0])
                # tmp_image = np.multiply(batch_targets[0][:,:,:4],255,dtype =np.uint8)
                # cv2.imshow("tmp_image", tmp_image)
                # cv2.waitKey()
                batch_inputs = np.reshape(batch_inputs,
                                          (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH,
                                           network.IMAGE_CHANNELS))
                batch_targets = np.reshape(batch_targets,
                                           (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH,
                                            network.IMAGE_INDEX))
                cost, _ = sess.run([network.cost, network.train_op],
                                   feed_dict={network.inputs: batch_inputs, network.targets: batch_targets,
                                              network.is_training: True})

                end = time.time()
                print('{}/{}, epoch: {}, cost: {}, batch time: {}'.format(batch_num,
                                                                          n_epochs * dataset.num_batches_in_epoch(),
                                                                          epoch_i, cost, end - start))

                if batch_num % 50 == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
                    test_inputs, test_targets = dataset.test_set()

                    test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 4))
                    test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 4))
                    # test_inputs = np.multiply(test_inputs, 1.0 / 255)

                    print(test_inputs.shape)
                    summary, test_accuracy = sess.run([network.summaries, network.accuracy],
                                                      feed_dict={network.inputs: test_inputs,
                                                                 network.targets: test_targets,
                                                                 network.is_training: False})

                    summary_writer.add_summary(summary, batch_num)

                    print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                    test_accuracies.append((test_accuracy, batch_num))
                    print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                    max_acc = max(test_accuracies)
                    print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))
                    if (test_accuracy >= max_acc[0]):
                        print("save model")
                        checkpoint_path = "save_model_New_human_5/model.ckpt"
                        saver.save(sess, checkpoint_path)
                    # Plot example reconstructions
                    # n_examples = 10
                    # test_inputs, test_targets = dataset.test_inputs[:n_examples], dataset.test_targets[:n_examples]
                    # test_inputs = np.multiply(test_inputs, 1.0 / 255)

                    # Prepare the plot

            if epoch_i % 10 == 0:
                print("check1")
                test_inputs, test_targets = dataset.test_set()

                # checkpoint_path = os.path.join('save', network.description, timestamp, 'model.ckpt')
                test_output_path = "save_model_New_human_5/save_model_output_" + str(epoch_i) + "/"

                test_segmentation = sess.run(network.segmentation_result, feed_dict={network.inputs: test_inputs})

                if not (util.check_folder(test_output_path)):
                    util.make_dir_folder(test_output_path)
                save_test_image(test_output_path, test_inputs, test_segmentation)

            """
            test_plot_buf = draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network,batch_num)

            # Convert PNG buffer to TF image
            image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)

            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            # Add image summary
            image_summary_op = tf.summary.image("plot", image)

            image_summary = sess.run(image_summary_op)
            summary_writer.add_summary(image_summary)
            """


if __name__ == '__main__':
    train()
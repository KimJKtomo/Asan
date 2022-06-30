import tensorflow as tf
tf.keras.backend.set_learning_phase(0)
import time
from keras.models import load_model
from Utils_model import VGG_LOSS
from keras.utils import plot_model
time.time()
image_shape = (64,32,3)
MODEL_PATH = "C:\\Users\\Hyeonseok\\Desktop\\dcscn-super-resolution-ver1\\SRGAN\\Keras-SRGAN\\Keras_to_TF"
## .h5 -> pb -> tensorlite
loss = VGG_LOSS(image_shape)
new_model = load_model('./model_2020_01_14/gen_model10000.h5', custom_objects={'vgg_loss': loss.vgg_loss},compile=True)
print(new_model.summary())


# tflite_converter = tf.lite.TFLiteConverter.from_keras_model_file(new_model)
# tflite_model = tflite_converter.convert()
# open("tf_lite_model.tflite", "wb").write(tflite_model)
# tf.train.write_graph({new_model}, "./Keras_to_TF/", "tf_model.pb", as_text=False)

sess = tf.keras.backend.get_session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

saver = tf.train.Saver()

save_path = saver.save(sess,MODEL_PATH)
print(save_path)
new_model.save(MODEL_PATH)
#
#
#
#
# print("Keras model is successfully converted to TF model in : "+MODEL_PATH )
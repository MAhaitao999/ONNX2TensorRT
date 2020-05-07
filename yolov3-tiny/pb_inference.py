import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


img1 = Image.open('../cat.jpg').resize((416, 416))
img1 = np.asarray(img1)
input_img = img1.reshape((1, 416, 416, 3))
print(input_img.shape)

# function to read ".pb" model
def read_pb_graph(model):
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


# perform inference using TensorFlow model
FROZEN_MODEL_PATH = "yolov3-tiny.pb"

graph = tf.Graph()
with graph.as_default():
    with tf.compat.v1.Session(config=config) as sess:
        # read TensorRT model
        frozen_graph = read_pb_graph(FROZEN_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(frozen_graph, name='')
        inputs = sess.graph.get_tensor_by_name('net1:0')
        outputs_1 = sess.graph.get_tensor_by_name('convolutional10/BiasAdd:0')
        outputs_2 = sess.graph.get_tensor_by_name('convolutional13/BiasAdd:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 10000
        out_pred = sess.run([outputs_1, outputs_2], feed_dict={inputs: input_img})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run([outputs_1, outputs_2], feed_dict={inputs: input_img})
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            print("needed time in inference-{}: {}ms".format(str(i), delta_time*1000))
        avg_time_original_model = total_time / n_time_inference
        print("average inference time: {}ms".format(avg_time_original_model*1000))


print(out_pred[0].shape)
print(out_pred[1].shape)

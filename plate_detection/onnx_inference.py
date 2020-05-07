import time
import onnx
import numpy as np
import onnxruntime
from PIL import Image


img = Image.open('../cat.jpg').resize((325, 325))
img = np.asarray(img)

input_tensor = img.reshape((1, 325, 325, 3)).astype(np.float32)
input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))
print(input_tensor.shape)

sess = onnxruntime.InferenceSession('plate_detection.onnx')
first_input_name = sess.get_inputs()[0].name
first_output_name = sess.get_outputs()[0].name
second_output_name = sess.get_outputs()[1].name

# input_tensor = np.random.rand(1, 3, 416, 416).astype(np.float32)
total_time = 0; n_time_inference = 10000
results = sess.run([first_output_name, second_output_name], {first_input_name: input_tensor})

for i in range(n_time_inference):
    t1 = time.time()
    results = sess.run([first_output_name, second_output_name], {first_input_name: input_tensor})
    t2 = time.time()
    delta_time = t2 - t1
    total_time += delta_time
    print('inference cost: {}ms'.format(delta_time*1000))

avg_time_original_model = total_time / n_time_inference
print("average inference time: {}ms".format(avg_time_original_model*1000))
print(results[0].shape)
print(results[1].shape)

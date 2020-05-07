import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import common


TRT_LOGGER = trt.Logger()


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def main():
    engine_file_path = 'yolov3-tiny.trt'
    input_image_path = '../cat.jpg'

    input_resolution_plate_detection_HW = (416, 416)
    preprocessor = PreprocessYOLO(input_resolution_plate_detection_HW)
    image_raw, image = preprocessor.process(input_image_path)
    print(image.shape)

    trt_outputs = []
    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image

        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 10000
        
        for i in range(n_time_inference):
            t1 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            print('inference-{} cost: {}ms'.format(str(i+1), delta_time*1000))
        avg_time_original_model = total_time / n_time_inference
        print("average inference time: {}ms".format(avg_time_original_model*1000))
        print(trt_outputs[0].shape)
        print(trt_outputs[1].shape)


if __name__ == '__main__':
    main()

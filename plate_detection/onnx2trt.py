import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES


import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import common


TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """
    Attempts to load a serialized engine if available,
    otherwise builds a new TensorRT engine and saves it.
    """
    print("hello world")
    def build_engine():
        """
        Takes an ONNX file and creates a TensorRT engine to run inference with
        """
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 2 << 30 # 1GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run platedetection2onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print(engine)
            print('Completed creating Engine')
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print('Reading engine from file {}'.format(engine_file_path))
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    """
    Create a TensorRT engine for ONNX-based plate_detection and run inference.
    """
    # Try to load a previously generated plate_detection graph in ONNX format:
    onnx_file_path = 'plate_detection.onnx'
    engine_file_path = 'plate_detection.trt'

    input_image_path = '../cat.jpg'

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_plate_detection_HW = (325, 325)
    preprocessor = PreprocessYOLO(input_resolution_plate_detection_HW)
    image_raw, image = preprocessor.process(input_image_path)
    print(image.shape)

    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        print('length of inputs is: ', len(inputs))
        print('inputs[0] is: \n', inputs[0])
        print('length of outputs is: ', len(outputs))
        print('outputs[0] is: \n', outputs[0])
        print('outputs[1] is: \n', outputs[1])

        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image

        for inp in inputs:
            print(inp.device)
            print(inp.host.shape)

        for i in range(100):
            t1 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
            t2 = time.time()
            print('inference cost: ', (t2 - t1)*1000, 'ms')
        print(trt_outputs[0].shape)
        print(trt_outputs[1].shape)


if __name__ == '__main__':
    main()

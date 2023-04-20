import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import time
import pyrealsense2 as rs
import sys

class DoFEngine(object):
    def __init__(self, engine_path, model_name, logger, print_log=False):
        self.model_name = model_name
        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,-1) # imagenet mean
        self.img_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,-1) # imagenet std
        self.logger =logger
        self.print_log = print_log

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        print("self.imgsz :", self.imgsz)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        # 여기에 nms 추가할 수는 없나
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        #self.logger.info("finished inference")
        data = [out['host'] for out in self.outputs]
        return data

    def preprocess(self, img):
        """
        preprocess: resized image and normalize (standardization) 
        input : cropped face img (h,w,c)
        output : resize and normalize image
        """
        if img.shape[0]==3:
            img = img.transpose(1,2,0) # (3, h, w) to (h, w, 3)
        img = cv2.resize(img, self.imgsz, interpolation=cv2.INTER_AREA)
        img = (img - self.img_mean) / self.img_std
        img = img.transpose(2,0,1) # (self.imgsz, self.imgsz, 3) to (3, self.imgsz, self.imgsz)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img

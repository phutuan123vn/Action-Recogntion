import time

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import threading

# from Yolov7 import Yolov7

def resize_img(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size
    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC
    # aspect ratio of image
    aspect = w/h
    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3
    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    return scaled_img


class myThread(threading.Thread):
   def __init__(self, func, args):
      threading.Thread.__init__(self)
      self.func = func
      self.args = args

   def run(self):
      print ("Starting thread" )
      print(self.func(self.args))
      print ("Exiting thread")


class Hrnet(object):

    def __init__(self, engine_path, imgsz=(256, 192)):
        self.imgsz = imgsz

        self.cfx = cuda.Device(0).make_context() #for multy thread

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


    def infer(self, img):
        self.cfx.push() # for multi thread
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        self.cfx.pop()
        return data

    def inference(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized, r, dwdh = self.letterbox(image, auto=False, new_shape=self.imgsz)
        img = image_resized.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255
        trt_outputs = self.infer(img)
        output_shapes = [(1, 17, 64, 48)]
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)][0]
        pose_keypoints, pose_score = self.process_output(trt_outputs, image_resized)
        pose_keypoints -= dwdh
        pose_keypoints /= r
        return pose_keypoints, pose_score


    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def process_output(self, heatmaps, image):
        img_height, img_width = image.shape[:2]
        # total_heatmap = cv2.resize(heatmaps.sum(axis=1)[0], (img_width, img_height))
        map_h, map_w = heatmaps.shape[2:]

        # Find the maximum value in each of the heatmaps and its location
        max_vals = np.array([np.max(heatmap) for heatmap in heatmaps[0, ...]])
        peaks = np.array([np.unravel_index(heatmap.argmax(), heatmap.shape)
                          for heatmap in heatmaps[0, ...]])
        # peaks[max_vals < conf_threshold] = np.array([0, 0])

        # Scale peaks to the image size
        peaks = peaks[:, ::-1] * np.array([img_width / map_w,
                                           img_height / map_h])
        # pose_result = np.column_stack((peaks, max_vals))
        return peaks, max_vals

    def inference_from_bbox(self, image, detections, search_region_ratio=0.2,score_threshold=0.5):
        pose_results = []
        full_height, full_width = image.shape[:2]
        boxes, scores, class_ids = detections
        if len(scores) == 0:
            return pose_results.append({'bbox':np.zeros(4), 'keypoints': np.zeros((17, 3))})
        person_class_idx = class_ids == 0
        boxes = boxes[person_class_idx]
        scores = scores[person_class_idx]
        if len(scores) == 0:
            return pose_results.append({'bbox':np.zeros(4), 'keypoints': np.zeros((17, 3))})
        # score = [x for x in ]
        area = []
        for (x1, y1, x2, y2) in boxes:
            area.append((x2-x1)*(y2-y1))
        idx_best = np.array(area).argmax()
        box = boxes[idx_best]
        score = scores[idx_best]
        x1, y1, x2, y2 = box
        box_width, box_height = x2 - x1, y2 - y1

        # Enlarge search region
        x1 = max(int(x1 - box_width * search_region_ratio), 0)
        x2 = min(int(x2 + box_width * search_region_ratio), full_width)
        y1 = max(int(y1 - box_height * search_region_ratio), 0)
        y2 = min(int(y2 + box_height * search_region_ratio), full_height)

        crop = image[y1:y2, x1:x2]
        if len(crop) == 0:
            return pose_results
        keypoint, kp_score = self.inference(crop)

        # Fix the body pose to the original image
        keypoint += np.array([x1, y1])

        pose_res = np.column_stack((keypoint, kp_score))
        box = list(box)
        box.append(score)
        det_res = np.array(box)
        pose_results.append({'bbox': det_res, 'keypoints': pose_res})

        return pose_results

    def destory(self):
        self.cfx.pop()

    def get_fps(self):
        # warmup
        import time
        img = np.ones((1, 3, self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(20):
            _ = self.infer(img)
        t1 = time.perf_counter()
        _ = self.infer(img)
        print(1 / (time.perf_counter() - t1), 'FPS')


def vis_pose(image, pose_result, threshold=0.3):
    bbox = []
    bbox_score = []
    keypoints = []
    keypoints_score = []
    for pos in pose_result:
        bbox.append(pos['bbox'][:4])
        bbox_score.append(pos['bbox'][4])
        keypoints.append(pos['keypoints'][:, :2])
        keypoints_score.append(pos['keypoints'][:, 2])
    max_score_indx = np.argmax(bbox_score)
    bbox = bbox[max_score_indx]
    keypoints = keypoints[max_score_indx]
    keypoints_score = keypoints_score[max_score_indx]
    skeleton_edge = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                     [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                     [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                     [3, 5], [4, 6]]
    for edge in skeleton_edge:
        start = keypoints[edge[0]]
        end = keypoints[edge[1]]
        if keypoints_score[edge[0]] < threshold or keypoints_score[edge[1]] < threshold:
            continue
        image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 255, 0), 2)

    for i in range(17):
        if keypoints_score[i] < threshold:
            continue
        (x, y) = keypoints[i]
        color = (255, 255, 255)

        image = cv2.circle(image, (int(x), int(y)), 10, color, -1)

    image_vis = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
    return image_vis


if __name__ == '__main__':
    Hrnet = Hrnet(engine_path='Pose/Hrnet48_fp16.trt')
    # Hrnet.get_fps()
    # Yolov7 = Yolov7(engine_path='Pose/yolov7_fp32.trt')
    # Yolov7.get_fps()
    # image = cv2.imread('Image/img_00160.jpg')
    # print(image.shape)

    # image = resize_img(image, (640, 480))
    # # image = Hrnet.letterbox(image, new_shape=(640, 480))[0]
    # # cv2.imshow('', image)
    # # cv2.waitKey(1)
    # # start = time.time()
    # detections = Yolov7.inference(image)
    # Yolov7.destory()
    # pose_results = Hrnet.inference_from_bbox(image, detections)
    # Hrnet.destory()
    # vis_image = vis_pose(image,[pose_results],0.5)
    # print(time.time() - start)
    # print(pose_results)
    # image_vis = vis_pose(image, pose_results)
    # print('end___')
    # cv2.imshow('res', vis_image)
    # cv2.waitKey(-1)

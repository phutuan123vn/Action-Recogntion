from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5 import uic
import imutils
import sys
import pandas as pd
import time
from Pose.Yolov7 import Yolov7
from Pose.Hrnet import Hrnet
import torch
from Model.model import ModelSTGCN
import cv2
import moviepy.editor as mpy
import mmcv.utils.progressbar as progressbar
# from mmdet.apis import inference_detector, init_detector
# from mmpose.apis import inference_top_down_pose_model, init_pose_model,vis_pose_result
import numpy as np
import shutil
import os.path as osp
import os
import glob
import torch.nn as nn
from copy import deepcopy


LABEL = [
    'Standing',
    'Walking',
    'Running',
    'Jump',
    'Fall',
    'Sitting',
    'Laying',
    'Get up'
]
FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def get_LabelAction(index:int)-> str: 
    return LABEL[index]

def resize_image(im, new_shape=(640, 480), color=(0, 0, 0), auto=False, scaleup=True, stride=32):
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
    return im


def inference_image(img,detect:Yolov7,pose:Hrnet,thr_det = 0.3,thr_kpt=0.3):
    det_results = detect.inference(img,threshold = thr_det)
    pose_results = pose.inference_from_bbox(img,det_results,score_threshold = thr_kpt)
    return pose_results

class Pose_detect_thread(QThread):
    progressing = pyqtSignal(int)
    pose_results = pyqtSignal(list)
    finished = pyqtSignal(float)   
    
    def __init__(self, det_model:Yolov7, pose_model:Hrnet,thr_det=0.5,thr_pose=0.5, video=None):
        super(Pose_detect_thread, self).__init__()
        self.det_model = det_model
        self.pose_model = pose_model
        self.video = video
        self.time_process = 0
        self.thr_det = thr_det
        self.thr_pose = thr_pose
    
    def run(self):
        start_time = time.time()
        cnt = 0
        pose_results = []
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flag, image = self.video.read()
        while flag:

            # image = resize_image(image)

            pose_res = inference_image(image,self.det_model,self.pose_model,self.thr_det,self.thr_pose)

            cnt += 1
            self.progressing.emit(cnt)
            pose_results.append(pose_res[0])
            flag, image = self.video.read()
        time_process = time.time() - start_time
        self.pose_results.emit(pose_results)
        self.finished.emit(time_process)

class My_GUI(QMainWindow):
    def __init__(self):
        super(My_GUI, self).__init__()
        uic.loadUi('Model.ui',self)
        self.show()
        ### CUSTOM FOR FOLDER IMAGE#################
        # self.dir_image = glob.glob('Image/*.png')
        # self.slider_frame_no.setRange(0, int(len(self.dir_image)) - 1)
        # self.slider_frame_no.setValue(0)
        ############################################
        self.msg = QMessageBox()
        self.keypoints = []
        self.msg.setWindowTitle('Error')
        # self.path_folder = osp.join('Image', 'img_{:05d}.jpg')
        # self.cnt=0
        self.Hrnet = Hrnet(engine_path='Pose/Hrnet48_fp32.trt')
        self.Hrnet.get_fps()
        self.Hrnet.destory()
        self.Yolov7 = Yolov7(engine_path='Pose/yolov7_fp16.trt')
        self.Yolov7.get_fps()
        self.Yolov7.destory()
        self.Model = ModelSTGCN(3,8)
        self.Model.load_state_dict(torch.load('Checkpoint/Model_best_wts Acc92.pt'))
        self.Model.to('cuda')
        self.Model.eval()
        # self.det_config = 'Pose/yolox_s_8x8_300e_coco.py'
        # self.det_checkpoint = 'Pose/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
        # self.pose_config = 'Pose/hrnet_w48_coco_256x192.py'
        # self.pose_checkpoint = 'Pose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

        self.skeleton_edge = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
                                (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                                (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                                (3, 5), (4, 6)]

        # self.label = None
        self.ano_lst = []
        self.action_label = None
        self.det_value = 0.5
        self.kpt_value = 0.5
        self.det_thr.setValidator(QDoubleValidator(0.00,1.00,2))
        self.kpt_thr.setValidator(QDoubleValidator(0.00,1.00,2))
        self.det_thr.editingFinished.connect(lambda widget = self.det_thr: self.changed(widget,0))
        self.kpt_thr.editingFinished.connect(lambda widget = self.kpt_thr: self.changed(widget,1))
        self.slider_frame_no.valueChanged.connect(self.frame_change)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_detect.clicked.connect(self.pose_detect)
        # self.btn_export.clicked.connect(self.export)
        self.btn_RUN.clicked.connect(self.Action_detect)
        self.thread = Pose_detect_thread(det_model=self.Yolov7, pose_model=self.Hrnet, video=None,thr_det = self.det_value,thr_pose=self.kpt_value)
        self.thread.progressing.connect(self.progressing_bar)
        self.thread.pose_results.connect(self.receive_pose_result)
        self.thread.finished.connect(self.print_time_pose_detect)
        
        
        
    def changed(self,widget,signal:int):
        # self.det_value = self.det_thr.text()
        # self.kpt_value = self.kpt_thr.text()
        # thr = float(thr)
        thr = widget.text()
        thr = float(thr) if thr != '' else 0.5
        if signal == 1: self.kpt_value = thr
        else: self.det_value = thr
        print(f'Signal: {signal}\n Thr:{thr}')
        
    def pose_detect(self):
        self.thread.video = self.Video
        self.thread.start()
        
    def progressing_bar(self, value):
        value = int(value * 100 / self.total_frame)
        self.progressBar.setValue(value)
        
    def receive_pose_result(self, pose_results):
        self.pose_results = pose_results

    def print_time_pose_detect(self, value):
        self.Text_time.setText(str(round(value, 4)))
        
    def load_image(self):
        self.image_path = QtWidgets.QFileDialog.getOpenFileName(self,'Open image file',filter='Image file (*.jpg *png)')[0]
        if len(self.image_path)==0:
            return
        self.frame_original= cv2.imread(self.image_path)
        # self.frame_original = cv2.resize(self.frame_original,(640,480))
        self.size_image = (self.frame_original.shape[1],self.frame_original.shape[0])
        print(f'Load image from: {self.image_path}')
        self.usingimage=True
        # self.det_value = self.det_thr.text()
        # self.kpt_value = self.kpt_thr.text()
        # if self.det_value =="":self.det_value=0.5
        # if self.kpt_value =="":self.kpt_value=0.5
        # print(f'DET THR: {self.det_value} \t KPT THR: {self.kpt_value}')
        # h,w = self.frame_original.shape[:2]
        # self.size_image = (w,h)
        self.frame_original.flags.writeable = False
        start = time.time()
        # _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
        #                                                   self.pose_checkpoint, self.frame_original)
        self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet,self.det_value,self.kpt_value)
        print(time.time() - start)
        if self.checkVis.isChecked():
            frame_show = self.vis_pose(self.frame_original, self.pose_result)
        else:
            bbox = self.pose_result['bbox']
            frame_show = cv2.rectangle(self.frame_original, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        self.current_frame = frame_show
        self.image_set(frame_show)

    def load_video(self):
        # self.Video_path = QtWidgets.QFileDialog.getOpenFileName()[0]
        # self.Video = cv2.VideoCapture(self.Video_path)
        # _, frame_show = self.Video.read()
        # self.size_image = frame_show.shape[:2]
        # self.image_set(frame_show)
        ###############
        self.Video_path = QtWidgets.QFileDialog.getOpenFileName(self,'Open video file', filter='Video files (*.mp4 *.mkv *.avi)')[0]
        if len(self.Video_path) == 0:
            return
        self.action_label = None
        print(f'Load Video: {self.Video_path}')
        self.usingimage=False
        # self.det_value = self.det_thr.text()
        # self.kpt_value = self.kpt_thr.text()
        # if self.det_value =="":self.det_value=0.5
        # if self.kpt_value =="":self.kpt_value=0.5
        self.Video = cv2.VideoCapture(self.Video_path)
        _, self.frame_original = self.Video.read()
        self.frame_original.flags.writeable = False
        self.frame_original = cv2.resize(self.frame_original,(640,480))
        self.size_image = (self.frame_original.shape[1],self.frame_original.shape[0])
        # self.image_set(self.frame_original)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(frame_show))
        self.total_frame = int(self.Video.get(cv2.CAP_PROP_FRAME_COUNT))
        # _, self.curr_frame = self.Video.read()
        self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet,self.det_value,self.kpt_value)
        if self.checkVis.isChecked():
            frame_show = self.vis_pose(self.frame_original, self.pose_result)
        else:
            bbox = self.pose_result[0]['bbox']
            frame_show = cv2.rectangle(self.frame_original, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        self.current_frame = frame_show
        self.image_set(frame_show)
        self.slider_frame_no.setRange(0, int(self.total_frame) - 1)
        self.slider_frame_no.setValue(0)
        # self.image_set(self.curr_frame)
    
    def frame_change(self, value):
        
        ######### CUSTOM FOR FOLDER IMAGE ############
        # self.frame_original = cv2.imread(self.dir_image[value])
        # self.frame_original.flags.writeable = False
        # image = resize_img(self.frame_original,(640,480))
        # self.Text_frame_no.setText(str(value))
        # # self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet)
        # self.pose_result = inference_image(image,self.Yolov7,self.Hrnet)
        # frame_show = self.vis_pose(image, self.pose_result)
        # self.image_set(frame_show)
        #############################################
        
        ######### MAIN ONE WHEN USE REMOVE ALL COMMENT ALL IN PARAGRAPH  ###########################
        self.Video.set(cv2.CAP_PROP_POS_FRAMES, value)
        _, self.frame_original = self.Video.read()
        self.frame_original = cv2.resize(self.frame_original,(640,480))
        # self.det_value = self.det_thr.text()
        # self.kpt_value = self.kpt_thr.text()
        # if self.det_value =="":self.det_value=0.5
        # if self.kpt_value =="":self.kpt_value=0.5
        # self.size_image = (self.frame_original.shape[1],self.frame_original.shape[0])
        # self.size_image = self.frame_original.shape[:2]
        # _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
        #                                                   self.pose_checkpoint, self.frame_original)
        # # print(time.time() - start)
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        self.frame_original.flags.writeable = False
        # self.frame_no = value
        self.Text_frame_no.setText(str(value))
        self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet,self.det_value,self.kpt_value)
        if self.checkVis.isChecked():
            frame_show = self.vis_pose(self.frame_original, self.pose_result)
        else:
            bbox = self.pose_result[0]['bbox']
            frame_show = cv2.rectangle(self.frame_original, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        self.current_frame = frame_show
        self.image_set(frame_show,value)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(self.frame_original))
        # self.image_set(frame)
        # self.frame_no_txt.setText(str(value))
        ###################################################
    
    def image_set(self, image,index:int=None):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if image.shape[0] > 640:
        #     # image = imutils.resize(image, height=640)
        #     image = resize_image(image,(640,480))
        # # image = resize_img(image,(640,480))
        # # self.size_image = image.shape[:2]
        # image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(image_Qt))
        ##########################
        if self.action_label is not None and self.usingimage == False:
            action = get_LabelAction(int(self.action_label[index]))
            try:
                xmin,ymin = self.pose_results[index]['bbox'][:2]
                coor = (int(xmin),int(ymin))
            except:
                pass
                # if self.pose_results[index-1] is not None:
                #     xmin,ymin = self.pose_results[index-1]['bbox'][:2]
                #     coor = (int(xmin),int(ymin))
            cv2.putText(image,action,coor, FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] > 640:
            # image = imutils.resize(image, height=640)
            image = resize_image(image,(640,480))
        # image = resize_img(image,(640,480))
        # self.size_image = image.shape[:2]
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.Label_Img_Show.setPixmap(QPixmap.fromImage(image_Qt))
            
    
    def vis_pose(self, image, pose_result):
        bbox = []
        bbox_score = []
        keypoints = []
        keypoints_score = []
        if pose_result is None:
            return image
        for pos in pose_result:
            bbox.append(pos['bbox'][:4])
            bbox_score.append(pos['bbox'][4])
            keypoints.append(pos['keypoints'][:,:2])
            keypoints_score.append(pos['keypoints'][:,2])
        max_score_indx = np.argmax(bbox_score)
        bbox = bbox[max_score_indx]
        keypoints = keypoints[max_score_indx]
        self.skeleton_features = pose_result[max_score_indx]['keypoints']
        self.keypoints = keypoints
        for edge in self.skeleton_edge:
            start = keypoints[edge[0]]
            end = keypoints[edge[1]]
            image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255,255,0), 2)
        for i in range(17):
            (x, y) = keypoints[i]
        #     if self.label[i] == 0:
        #         color = (255, 255, 255)
        #     elif self.label[i] == 1:
        #         color = (0, 0, 255)
        #     elif self.label[i] == 2:
        #         color = (255, 0, 0)
            image = cv2.circle(image, (int(x), int(y)), 4, (255, 255, 255), -1)

        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        return image
    
    def Action_detect(self):
        lst_kp = []
        for pose in self.pose_results:
            lst_kp.append(deepcopy(pose['keypoints']))
        kp = np.array(lst_kp)
        w,h = self.size_image
        kp[:,:,0] = (kp[:,:,0]-w/2)/(w/2)
        kp[:,:,1] = (kp[:,:,1]-h/2)/(h/2)
        Action_window = np.zeros((1,len(self.pose_results),1))
        for window in range(0,len(self.pose_results),15):
            feature = np.expand_dims(kp[window:window+30],0)
            temp = torch.from_numpy(feature).float().to('cuda')
            outputs = self.Model(temp)
            pred = torch.argmax(outputs,1)
            Action_window[0][window:window+30] = pred.item()
        self.action_label = Action_window[0]
        print(self.action_label)
        print(kp)
        
        
def main():
    app = QApplication([])
    window = My_GUI()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
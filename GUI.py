from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
import imutils
import numpy as np
import copy
import sys
import pandas as pd
import time
import os.path as osp
from inference_topdown_pose import inference_img

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

class My_GUI(QMainWindow):
    def __init__(self):
        super(My_GUI, self).__init__()
        uic.loadUi('form1.ui',self)
        self.show()
        self.msg = QMessageBox()
        self.keypoints = []
        self.msg.setWindowTitle('Error')
        self.path_folder = osp.join('Image', 'img_{:05d}.jpg')
        self.cnt=0
        self.det_config = 'Pose/yolox_s_8x8_300e_coco.py'
        self.det_checkpoint = 'Pose/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
        self.pose_config = 'Pose/hrnet_w48_coco_256x192.py'
        self.pose_checkpoint = 'Pose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

        self.skeleton_edge = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
                                (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                                (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                                (3, 5), (4, 6)]

        # self.label = None
        self.ano_lst = []
        self.anno = []
        self.slider_frame_no.valueChanged.connect(self.frame_change)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save)
        self.btn_export.clicked.connect(self.export)
        self.btn_detect_vid.clicked.connect(self.detect)
        self.btn_remove_last.clicked.connect(self.remove)
        self.btn_Append.clicked.connect(self.Add_anno)
    
    def Add_anno(self):
        return
    
    def remove(self):
        if len(self.ano_lst)==0 or len(self.anno == 0):
            return
        self.ano_lst.pop()
        self.anno.pop()
        self.Text_display.setText(f'len ano: {len(self.ano_lst)} \nsave:{self.ano_lst}')
    
    def detect(self):
        ## detect pose
        # start = time.time()
        # self.frame_original.flags.writeable = False
        # self.size_image = self.frame_original.shape[:2]
        # _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
        #                                                   self.pose_checkpoint, self.frame_original)
        # print(time.time() - start)
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        # self.image_set(frame_show)
        #################
        # if not self.usingimage:
        #     return
        self.frame_no +=1
        frame_path = self.path_folder.format(self.cnt + 1)
        self.cnt +=1
        cv2.imwrite(frame_path, self.frame_original)
        self.slider_frame_no.setValue(self.cnt)
        self.frame_change(self.frame_no)
        

    def load_image(self):
        self.image_path = QtWidgets.QFileDialog.getOpenFileName(self,'Open image file',filter='Image file (*.jpg *png)')[0]
        if len(self.image_path)==0:
            return
        self.frame_original= cv2.imread(self.image_path)
        print(f'Load image from: {self.image_path}')
        self.usingimage=True
        self.size_image = self.frame_original.shape[:2]
        self.frame_original.flags.writeable = False
        start = time.time()
        _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
                                                          self.pose_checkpoint, self.frame_original)
        print(time.time() - start)
        frame_show = self.vis_pose(self.frame_original, self.pose_result)
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
        self.usingimage=False
        self.Video = cv2.VideoCapture(self.Video_path)
        _, self.frame_original = self.Video.read()
        self.frame_original.flags.writeable = False
        self.image_set(self.frame_original)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(frame_show))
        self.total_frame = int(self.Video.get(cv2.CAP_PROP_FRAME_COUNT))
        _, self.curr_frame = self.Video.read()
        self.slider_frame_no.setRange(0, int(self.total_frame) - 1)
        self.slider_frame_no.setValue(0)
        # self.image_set(self.curr_frame)
    
    def frame_change(self, value):
        self.Video.set(cv2.CAP_PROP_POS_FRAMES, value)
        _, self.frame_original = self.Video.read()
        # self.size_image = self.frame_original.shape[:2]
        # _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
        #                                                   self.pose_checkpoint, self.frame_original)
        # # print(time.time() - start)
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        self.frame_original.flags.writeable = False
        self.frame_no = value
        self.image_set(self.frame_original)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(self.frame_original))
        # self.image_set(frame)
        # self.frame_no_txt.setText(str(value))

    def vis_pose(self, image, pose_result):
        bbox = []
        bbox_score = []
        keypoints = []
        keypoints_score = []
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


    def image_set(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] >640:
            image = imutils.resize(image, height=640)
        # image = resize_img(image,(640,480))
        # self.size_image = image.shape[:2]
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.Label_Img_Show.setPixmap(QPixmap.fromImage(image_Qt))

    def save(self):
        self.label = self.Label_Edit.text()
        if self.label == '':
            self.msg.setText('Please input a label to save')
            self.msg.exec_()
        else:
            self.anno.append(self.skeleton_features)
            self.ano_lst.append({'keypoints': self.skeleton_features, 'label':self.label, 'image size': self.size_image})
            if self.usingimage:
                print(f'Save image from: {self.image_path}')
            else:
                self.frame_change(self.frame_no)
            self.Text_display.setText(f'len ano: {len(self.ano_lst)} \nsave:{self.ano_lst}')

    def export(self):
        self.label = self.Label_Edit.text()
        file_name = self.Edit_file_name.text()
        if self.label == '' or file_name == '':
            self.msg.setText('Please input a label and path to save')
            self.msg.exec_()
        else:
            pd.to_pickle(self.anno, 'Data/pickle_file/'+file_name)

def main():
    app = QApplication([])
    window = My_GUI()
    app.exec_()


if __name__ == "__main__":
    main()
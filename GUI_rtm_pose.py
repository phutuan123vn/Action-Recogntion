from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from mmdeploy_runtime import PoseDetector,Detector
import cv2
import imutils
from copy import deepcopy
import numpy as np
# import copy
import sys
import pandas as pd
import time
import os.path as osp
import glob
from Pose.Hrnet import Hrnet
from Pose.Yolov7 import Yolov7
det_model = Detector('Pose-Estimate/Yolov7-tiny/','cuda')
pose_model = PoseDetector('Pose-Estimate/HRNet/','cuda')

def visualize(frame, anno:dict, thr=0.5, resize=None):
    if anno is None:
        if resize:
            scale = resize / max(frame.shape[0], frame.shape[1])
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        return frame
    keypoints = anno['keypoints']
    bboxes = anno['bbox']
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    palette = [(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
               (255, 153, 255), (153, 204, 255), (255, 102, 255),
               (255, 51, 255), (102, 178, 255),
               (51, 153, 255), (255, 153, 153), (255, 102, 102), (255, 51, 51),
               (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
               (0, 0, 255), (255, 0, 0), (255, 255, 255)]
    link_color = [
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]
    point_color = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]
    img = deepcopy(frame)
    scores = keypoints[..., 2]
    keypoints = (keypoints[..., :2]).astype(int)
    if resize:
        scale = resize / max(frame.shape[0], frame.shape[1])

       
        keypoints = (keypoints[..., :2] * scale).astype(int)
        for i,b in enumerate(bboxes):
            bboxes[i] = b*scale
        img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    for kpts, score,bbox in zip(keypoints, scores, bboxes):
        show = [0] * len(kpts)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, tuple(kpts[u]), tuple(kpts[v]), palette[color], 1,
                         cv2.LINE_AA)
                show[u] = show[v] = 1
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
               cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
    return img

def inference_image(img,thr=0.2):
    bboxes, labels, _ = det_model(img)
    keep = np.logical_and(labels == 0, bboxes[..., 4] > thr)
    bboxes = bboxes[keep]
    if len(bboxes) == 0:
        return None
    bbox = bboxes[0,:4]
    score_det = bboxes[0,4]
    keypoints = pose_model(img,bbox)
    anno = dict()
    anno['bbox'] = [bbox]
    anno['keypoints'] = keypoints
    return anno
    
def inference_image_multi(img,thr=0.7):
    bboxes, labels, _ = det_model(img)
    keep = np.logical_and(labels == 0, bboxes[..., 4] > thr)
    bboxes = bboxes[keep]
    if len(bboxes) == 0:
        return None
    bbox = []
    score_det = []
    for item in bboxes:
        bbox.append(item[...,:4])
        score_det.append(item[4])
    keypoints = pose_model(img,bbox)
    anno = dict()
    anno['bbox'] = bbox
    anno['keypoints'] = keypoints
    return anno

    
class My_GUI(QMainWindow):
    def __init__(self):
        super(My_GUI, self).__init__()
        uic.loadUi('form1.ui',self)
        self.show()
        ### CUSTOM FOR FOLDER IMAGE#################
        # self.dir_image = glob.glob('Image/*.png')
        # self.slider_frame_no.setRange(0, int(len(self.dir_image)) - 1)
        # self.slider_frame_no.setValue(0)
        ############################################
        self.msg = QMessageBox()
        self.keypoints = []
        self.msg.setWindowTitle('Error')
        ### model
        # self.path_folder = osp.join('Image', 'img_{:05d}.jpg')
        # self.Hrnet = Hrnet(engine_path='Pose/Hrnet48_fp32.trt')
        # self.Hrnet.get_fps()
        # self.Hrnet.destory()
        # self.Yolov7 = Yolov7(engine_path='Pose/yolov7_fp16.trt')
        # self.Yolov7.get_fps()
        # self.Yolov7.destory()
        # self.det_config = 'Pose/yolox_s_8x8_300e_coco.py'
        # self.det_checkpoint = 'Pose/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
        # self.pose_config = 'Pose/hrnet_w48_coco_256x192.py'
        # self.pose_checkpoint = 'Pose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        ####

        self.skeleton_edge = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
                                (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                                (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                                (3, 5), (4, 6)]

        # self.label = None
        self.ano_lst = []
        self.anno = []
        # self.image_size = []
        # self.lst_size = []
        # self.skeleton_features
        # self.LineEdit.setValidator()
        self.Label_Edit.setValidator(QIntValidator())
        self.slider_frame_no.valueChanged.connect(self.frame_change)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save)
        self.btn_export.clicked.connect(self.export)
        self.btn_detect_vid.clicked.connect(self.detect)
        self.btn_remove_list.clicked.connect(self.remove_list)
        self.btn_tolist.clicked.connect(self.save_action)
        self.btn_remove_frame.clicked.connect(self.remove_frame)
    
    def save_action(self):
        self.label = self.Label_Edit.text()
        frame = len(self.anno)
        if self.label == '' or frame == 0:
            self.msg.setText('Missing Frame or Label to save')
            self.msg.exec_()
        else:
            pose = np.zeros((1,frame,17,3),dtype=np.float32)
            for idex, anno in enumerate(self.anno):
                pose[0][idex] = anno
            temp = dict()
            temp['kp'] = pose
            temp['label'] = self.label
            temp['img_shape'] = self.size_image
            temp['total_frame'] = frame
            self.anno = []
            self.ano_lst.append(temp)
            self.Text_display.setText(f'Length anno: {len(self.ano_lst)} \nsave:{self.ano_lst}')
    
    def remove_frame(self):
        if len(self.anno) == 0:
            return
        self.anno.pop()
        self.Text_display.setText(f'Length Frame: {len(self.anno)} \nsave:{self.anno}')
        
    def remove_list(self):
        if len(self.ano_lst) == 0:
            return
        self.ano_lst.pop()
        self.Text_display.setText(f'Length anno: {len(self.ano_lst)} \nsave:{self.ano_lst}')
    
    def detect(self):
        ## detect pose
        # start = time.time()
        if self.usingimage:
            return
        # self.size_image = self.frame_original.shape[:2]
        self.frame_original.flags.writeable = False
        bboxes, labels, _ = det_model(self.frame_original)
        keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.2)
        bboxes = bboxes[keep]
        print(f'BBOX: {bboxes}')
        if len(bboxes) == 0:
            return 
        bbox = bboxes[0,:4]
        img  = cv2.rectangle(self.frame_original, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        self.image_set(img)
        # pose_result = inference_image(self.frame_original,thr=0.5)
        # self.skeleton_features = pose_result['keypoints']
        # frame_show = self.vis_pose(self.frame_original, pose_result)
        # self.image_set(frame_show)
        #####################
        
        #################
        # self.frame_no +=1
        # frame_path = self.path_folder.format(self.cnt + 1)
        # self.cnt +=1
        # cv2.imwrite(frame_path, self.frame_original)
        # self.slider_frame_no.setValue(self.cnt)
        # self.frame_change(self.frame_no)
        

    def load_image(self):
        self.image_path = QtWidgets.QFileDialog.getOpenFileName(self,'Open image file',filter='Image file (*.jpg *png)')[0]
        if len(self.image_path)==0:
            return
        self.frame_original= cv2.imread(self.image_path)
        print(f'Load image from: {self.image_path}')
        self.usingimage=True
        h,w = self.frame_original.shape[:2]
        self.size_image = (w,h)
        self.frame_original.flags.writeable = False
        start = time.time()
        # _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
        #                                                   self.pose_checkpoint, self.frame_original)
        # self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet)
        pose_result = inference_image(self.frame_original)
        # self.skeleton_features = pose_result['keypoints']
        print(time.time() - start)
        frame_show = self.vis_pose(self.frame_original, pose_result)
        # frame_show = visualize(self.frame_original,pose_result,resize=640)
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
        print(f'Load Video: {self.Video_path}')
        self.usingimage=False
        self.Video = cv2.VideoCapture(self.Video_path)
        _, self.frame_original = self.Video.read()
        self.size_image = (self.frame_original.shape[1],self.frame_original.shape[0])
        self.frame_original.flags.writeable = False
        # self.image_set(self.frame_original)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(frame_show))
        self.total_frame = int(self.Video.get(cv2.CAP_PROP_FRAME_COUNT))
        # _, self.curr_frame = self.Video.read()
        #####################
        pose_result = inference_image(self.frame_original)
        frame_show = self.vis_pose(self.frame_original, pose_result)
        # self.skeleton_features = pose_result['keypoints']
        self.image_set(frame_show)
        ###################
        # self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet)
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        # self.image_set(frame_show)
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
        # self.size_image = self.frame_original.shape[:2]
        # _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
        #                                                   self.pose_checkpoint, self.frame_original)
        # # print(time.time() - start)
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        self.frame_original.flags.writeable = False
        # self.frame_no = value
        self.Text_frame_no.setText(str(value))
        # pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet)
        pose_result = inference_image(self.frame_original)
        frame_show = self.vis_pose(self.frame_original, pose_result)
        # self.skeleton_features = pose_result['keypoints']
        self.image_set(frame_show)
        ############################
        # self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet)
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        # self.image_set(frame_show)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(self.frame_original))
        # self.image_set(frame)
        # self.frame_no_txt.setText(str(value))
        ###################################################
        
    def vis_pose(self, image, pose_result,resize=640):
        keypoints = []
        img = visualize(image,pose_result,resize=resize)
        if pose_result is None:
            return img
        self.skeleton_features = pose_result['keypoints']
        return img
        # self.skeleton_features = pose_result['keypoints']
        # bbox = []
        # bbox_score = []
        # keypoints = []
        # keypoints_score = []
        # if pose_result is None:
        #     return image
        # for pos in pose_result:
        #     bbox.append(pos['bbox'][:4])
        #     bbox_score.append(pos['bbox'][4])
        #     keypoints.append(pos['keypoints'][:,:2])
        #     keypoints_score.append(pos['keypoints'][:,2])
        # max_score_indx = np.argmax(bbox_score)
        # bbox = bbox[max_score_indx]
        # keypoints = keypoints[max_score_indx]
        # self.skeleton_features = pose_result[max_score_indx]['keypoints']
        # self.keypoints = keypoints
        # for edge in self.skeleton_edge:
        #     start = keypoints[edge[0]]
        #     end = keypoints[edge[1]]
        #     image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255,255,0), 2)
        # for i in range(17):
        #     (x, y) = keypoints[i]
        # #     if self.label[i] == 0:
        # #         color = (255, 255, 255)
        # #     elif self.label[i] == 1:
        # #         color = (0, 0, 255)
        # #     elif self.label[i] == 2:
        # #         color = (255, 0, 0)
        #     image = cv2.circle(image, (int(x), int(y)), 4, (255, 255, 255), -1)

        # image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        # return image


    def image_set(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] > 640:
            image = imutils.resize(image, height=640)
            # image = resize_img(image,(640,480))
        # image = resize_img(image,(640,480))
        # self.size_image = image.shape[:2]
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.Label_Img_Show.setPixmap(QPixmap.fromImage(image_Qt))

    def save(self):
        # self.label = self.Label_Edit.text()
        # if self.label == '':
        #     self.msg.setText('Please input a label to save')
        #     self.msg.exec_()
        # else:
        if len(self.skeleton_features) == 0:
            self.msg.setText(f"Frame don't have the skeleton" )
            self.msg.exec_()
        self.anno.append(self.skeleton_features)
        # self.ano_lst.append({'keypoints': self.skeleton_features, 'label':self.label, 'image size': self.size_image})
        if self.usingimage:
            print(f'Save image from: {self.image_path}')
        else:
            self.frame_no = self.slider_frame_no.value() + 1
            self.slider_frame_no.setValue(self.frame_no)
            self.frame_change(self.frame_no)
        self.Text_display.setText(f'Length Frame: {len(self.anno)} \nsave:{self.anno}')

    def export(self):
        file_name = self.Edit_file_name.text()
        if file_name == '':
            self.msg.setText('Please input a file name to save')
            self.msg.exec_()
        else:
            pd.to_pickle(self.ano_lst, 'Data/pickle_file/'+file_name)

def main():
    app = QApplication([])
    window = My_GUI()
    app.exec_()


if __name__ == "__main__":
    main()
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


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

pose_model = Hrnet(engine_path='Pose/Hrnet48_fp32.trt')
pose_model.get_fps()
pose_model.destory()
det_model = Yolov7(engine_path='Pose/yolov7_fp16.trt')
det_model.get_fps()
det_model.destory()
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

skeleton_edge = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
                                (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                                (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                                (3, 5), (4, 6)]

def inference_image(img,detect:Yolov7,pose:Hrnet):
    det_results = detect.inference(img)
    pose_results = pose.inference_from_bbox(img,det_results)
    return pose_results

def get_LabelAction(index:int)-> str: 
    return LABEL[index]

def vis_pose(image, pose_result,threshold = 0.5):
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
        keypoints_score = keypoints_score[max_score_indx]
        skeleton_features = pose_result[max_score_indx]['keypoints']
        keypoints = keypoints
        for edge in skeleton_edge:
            start = keypoints[edge[0]]
            end = keypoints[edge[1]]
            # image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255,255,0), 2)
            if keypoints_score[edge[0]] < threshold or keypoints_score[edge[1]] < threshold:
                continue
            image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 255, 0), 2)
        for i in range(17):
            (x, y) = keypoints[i]
            if keypoints_score[i] < threshold:
                continue
        #     if self.label[i] == 0:
        #         color = (255, 255, 255)
        #     elif self.label[i] == 1:
        #         color = (0, 0, 255)
        #     elif self.label[i] == 2:
        #         color = (255, 0, 0)
            image = cv2.circle(image, (int(x), int(y)), 4, (255, 255, 255), -1)

        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        return image

def extract_frame(video_path):
    dname = 'temp'
    os.makedirs(dname, exist_ok=True)
    frame_tmpl = osp.join(dname, 'img_{:05d}.jpg')
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    cnt = 0
    while(cap.isOpened()):
        flag, frame = cap.read()
        if flag:
            frame_path = frame_tmpl.format(cnt + 1)
            frame_paths.append(frame_path)
            frame=cv2.resize(frame,(640,480))
            cv2.imwrite(frame_path, frame)
            cnt += 1
        else: break
    cap.release()
    return frame_paths

def detection_inference(det_model:Yolov7,frame_paths,det_score=0.5):
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = progressbar.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        result = det_model.inference(img,det_score)
        # We only keep human detections with score larger than det_score_thr
        if len(result[2]) == 0:
            results.append(result)
            prog_bar.update()
            continue
        person_id = result[2] == 0
        bbox = result[0][person_id]
        score = result[1][person_id]
        indx = result[2][result[2]==0]
        results.append((bbox,score,indx))
        prog_bar.update()
    print('\n')
    return results

def pose_inference(pose_model:Hrnet,frame_paths,det_results):
    print('Performing Human Pose Estimation for each frame')
    prog_bar = progressbar.ProgressBar(len(frame_paths))
    num_frame = len(det_results)
    num_person = max([len(x[2]) for x in det_results])
    if num_person == 0:
        kp = np.zeros((1,num_frame,17,3),dtype=np.float32)
        return kp
    kp = np.zeros((num_person,num_frame,17,3))
    pose_info = []
    vis_frames = []
    for i ,(f,d) in enumerate(zip(frame_paths,det_results)):
        img = cv2.imread(f)
        pose_result = pose_model.inference_from_bbox(img,d)
        if pose_result is None:
            pose_info.append([])
            for person_id in range(num_person):
                kp[person_id,i] = kp[person_id,i-1]
        else:
            pose_info.append(pose_result)
            for j,item in enumerate(pose_result):
                normkp = deepcopy(item["keypoints"])
                normkp = normalize_kp(normkp,(img.shape[1],img.shape[0]))
                kp [j,i] = normkp
        vis_image = vis_pose(img,pose_result)
        vis_frames.append(vis_image)
        # cv2.imshow('',vis_image)
        # if cv2.waitKey(20)& 0xFF==ord('q'): break
        prog_bar.update()
    print('\n')
    cv2.destroyAllWindows()
    return kp,pose_info,vis_frames

def pose_extraction(vid,label,pose_model:Hrnet=pose_model,det_model:Yolov7=det_model,det_score=0.5):
    frame_paths = extract_frame(vid)
    det_results = detection_inference(det_model,frame_paths,det_score)
    img = cv2.imread(frame_paths[0])
    img_shape = (img.shape[1],img.shape[0])
    pose_results = pose_inference(pose_model,frame_paths,det_results)
    anno = dict()
    anno['kp'] = pose_results
    anno['img_shape'] = img_shape
    anno['total_frames'] = pose_results.shape[1]
    anno['label'] = label
    shutil.rmtree(osp.dirname(frame_paths[0]))
    return anno

def extend_bbox(pose_results):
    bbox = []
    for person in pose_results:
        xmin,ymin = person['bbox'][:2]
        box = tuple(xmin,ymin)
        
        
def normalize_kp(kp,image_shape):
    w,h = image_shape
    kp[:,0] = (kp[:,0]-w/2)/(w/2)
    kp[:,1] = (kp[:,1]-h/2)/(h/2)
    return kp


def ActionReg(model:nn.Module = None,file:str = None,det_score = 0.5,ouput_name:str = 'Out.mp4',
              det_model:Yolov7 = None, pose_model:Hrnet = None,device = 'cuda'):
    assert all(param is not None for param in [model,file,det_model,pose_model]),"All param must be give in"
    model.to(device)
    frame_paths = extract_frame(file)
    det_results = detection_inference(det_model,frame_paths,det_score)
    img = cv2.imread(frame_paths[0])
    img_shape = (img.shape[1],img.shape[0])
    pose_results,pose_frame,vis_images = pose_inference(pose_model,frame_paths,det_results)
    Action_window = np.zeros((pose_results.shape[0],pose_results.shape[1],1))
    for window in range(0,len(frame_paths),15):
        feature = np.expand_dims(pose_results[0][window:window+30],0)
        temp = torch.from_numpy(feature).float().to(device)
        outputs = model(temp)
        pred = torch.argmax(outputs,1)
        Action_window[0][window:window+30] = pred.item()
    action_label = Action_window[0]
    for index,frame in enumerate(vis_images):
        # if  action_label[index] == 1:
        #     action = 'Fall'
        # elif action_label[index] == 0:
        #     action = 'Not Fall'
        # else: action = 'No action'
        action = get_LabelAction(int(action_label[index]))
        try:
            xmin,ymin = pose_frame[index][0]['bbox'][:2]
            coor = (int(xmin),int(ymin))
        except:
            continue
        cv2.putText(frame,action,coor, FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_images], fps=24)
    vid.write_videofile(ouput_name, remove_temp=True)
    print('Success writing Video')

if __name__=='__main__': 
    model = ModelSTGCN(3,8)
    model.load_state_dict(torch.load('Checkpoint/Model_best_wts Acc92.pt'))
    model.eval()
    # file = "Train & Val/VAL/WALK/17.mp4"
    # output_name = "Video/OutWalk17.mp4"
    file = 'Video/trip1.mp4'
    output_name = 'Video/OutTrip1.mp4'
    model.cuda()
    ActionReg(model=model,file=file,det_score=0.5,
              det_model=det_model,pose_model=pose_model,ouput_name=output_name)
    # ActionReg(model,file,0.8,'Out1.mp4',det_model,pose_model)


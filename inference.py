import os
import sys
import argparse
from typing import OrderedDict
from loguru import logger
sys.path.append(os.getcwd())
import onnxruntime
import cv2
import glob
import json
from tqdm import tqdm
import time
import numpy as np
from attrdict import AttrDict
import torch, torchvision
import albumentations as A
from PIL import ImageFont, ImageDraw, Image

detector_list = ['SSD', 'RetinaNet', 'YOLOv5', "SSD4Point"]
classifier_list = ['Classifier']
ocr_list        = ['BaseLpr']

def json_to_dict(json_path):
    with open(json_path) as json_file:
        return json.load(json_file)


def putText(img, text, org, font_path, color=(0, 0, 255), font_size=20):
    """
    Display text on images
    :param img: Input img, read through cv2
    :param text: 표시할 텍스트 
    :param org: The coordinates of the upper left corner of the text
    :param font_path: font path
    :param color: font color, (B,G,R)
    :return:
    """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    b, g, r = color
    a = 0
    draw.text(org, text, stroke_width=2, font=ImageFont.truetype(font_path, font_size), fill=(b, g, r, a))
    img = np.array(img_pil)
    return img


def nms(results, detections_per_img, nms_thresh, min_score=0.05):
    detections = []
    for batch_res in  results:
        batch_boxes, batch_scores, batch_labels = batch_res
        for image_boxes, image_scores, image_labels in zip(batch_boxes, batch_scores, batch_labels):
            if image_boxes[0].shape[0] == 8:
                image_boxes_2point = image_boxes[:,[0,1,4,5]]
                index = image_scores > min_score
                keep = torchvision.ops.batched_nms(image_boxes_2point[index], image_scores[index], image_labels[index], nms_thresh)
                keep = keep[:detections_per_img]
            else:
                index = image_scores > min_score
                keep = torchvision.ops.batched_nms(image_boxes[index], image_scores[index], image_labels[index], nms_thresh)
                keep = keep[:detections_per_img]
            detections.append({
                        'boxes': image_boxes[index][keep],
                        'scores': image_scores[index][keep],
                        'labels': image_labels[index][keep],
                    })
    return detections

baseLpr_dict = {"rk":"가", "sk":"나", "ek":"다", "fk":"라", "ak":"마", "qk":"바", "tk":"사", "dk":"아", "wk":"자",
                    "rj":"거", "sj":"너", "ej":"더", "fj":"러", "aj":"머", "qj":"버", "tj":"서", "dj":"어", "wj":"저", 
                    "rh":"고", "sh":"노", "eh":"도", "fh":"로", "ah":"모", "qh":"보", "th":"소", "dh":"오", "wh":"조", 
                    "rn":"구", "sn":"누", "en":"두", "fn":"루", "an":"무", "qn":"부", "tn":"수", "dn":"우", "wn":"주", 
                    "gj":"허", "gk":"하", "gh":"호", "a":"서울", "b":"경기", "c":"인천", "d":"강원", "e":"충남",
                    "f":"대전", "g":"충북", "h":"부산", "i":"울산", "j":"대구", "k":"경북", "l":"경남", "m":"전남", "n":"광주",
                    "o":"전북", "p":"제주", "세종": "세종", "임":"임", "크":"크","0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6",
                "7":"7", "8":"8", "9":"9", "[s]":"[s]", "[GO]":"[GO]", "배":"배"}
car_maker = ["Hyundai", "Kia", "Audi", "Tesla", "Chevrolet", "BMW", "Ford", "Honda",
            "Infiniti", "Jeep", "Lexus", "Nissan" ,  "Volkswagen", 
            "SSangyong", "TOYOTA", "VOLVO", "RENAULT" , "Mercedes-Benz" , "Porsche",
            "PEUGEOT", "JAGUAR", "LANDROVER", "Mercedes-Maybach", "Genesis", "Chrysler",
            "Bentley", "Cadillac", "LINCOLN", "Mini", "Discovery", "GM DAEWOO", "Alpheon",
            "Maserati", "EQUUS", "Stinger", "Opirus", "Mohave", "Citroen", "Mazda", "Buick"]
character = "[blank],0,1,2,3,4,5,6,7,8,9,rk,sk,ek,fk,ak,qk,tk,dk,wk,rj,sj,ej,fj,aj,qj,tj,dj,wj,rh,sh,eh,fh,ah,qh,th,dh,wh,rn,sn,en,fn,an,qn,tn,dn,wn,gj,gk,gh,d,b,l,k,n,j,f,h,a,세종,i,c,m,o,p,e,g,배,크,임".split(",")

def make_parser():
    parser = argparse.ArgumentParser("DYIOT App")
 
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--sub_ckpt1", default=None, type=str, help="sub model's checkpoint file")
    parser.add_argument("--sub_ckpt2", default=None, type=str, help="sub model's checkpoint file")
    parser.add_argument("--inference_method", default='Image', type=str, help="Input source to infer: Image | Video | Stream")
    parser.add_argument("--data_cfg", default='./data.config', type=str, help="data config for onnx inference")
    return parser

def get_batch(image_l):
    return np.stack(image_l, 0)
    
def pad_and_reshape(labels, ratio, pad, shape):
        (h, w) = shape        
        if len(labels) > 0:
            # Normalized 4point pixel xyxy format
            x = labels.copy()
            labels[ 0] = ratio[0] * w * (x[ 0]) + pad[0]
            labels[ 1] = ratio[1] * h * (x[ 1]) + pad[1] 
            labels[ 2] = ratio[0] * w * (x[ 2]) + pad[0]  
            labels[ 3] = ratio[1] * h * (x[ 3]) + pad[1]
            labels[ 4] = ratio[0] * w * (x[ 4]) + pad[0]
            labels[ 5] = ratio[1] * h * (x[ 5]) + pad[1]  
            labels[ 6] = ratio[0] * w * (x[ 6]) + pad[0] 
            labels[ 7] = ratio[1] * h * (x[ 7]) + pad[1]
        return labels

def decode( text_index, length):
    """ convert text-index into text-label. """
    texts = []
    index = 0
    for l in length:

        t = text_index[index:index + l]
        char_list = []
        for i in range(l):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                char_list.append(baseLpr_dict[character[t[i]]])
        text = ''.join(char_list)
        texts.append(text)
        index += l
    return texts

def pre_process(img,  img_size):

    img = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_LINEAR)

    augmentation = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255, always_apply=True)], bbox_params=A.BboxParams(format='pascal_voc'))
    transformed = augmentation(image=img, bboxes=np.array([]))
    img = transformed["image"]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    img = get_batch([img])

    return img

def lpr_pre_process(ori_img, bbox, img_shape):
    (h,w) = img_shape
    margin_x = int((bbox[4] - bbox[0])/2)
    margin_y = int((bbox[5] - bbox[1])/2)
    roi_corners = np.array([[(bbox[0]-margin_x, bbox[1]-margin_y), (bbox[2]+margin_x, bbox[3]-margin_y), (bbox[4]+margin_x, bbox[5]+margin_y), (bbox[6]-margin_x, bbox[7]+margin_y)]], dtype=np.int32)
    pts1 = np.float32([[bbox[0]-margin_x, bbox[1]-margin_y], [bbox[2]+margin_x, bbox[3]-margin_y], [bbox[4]+margin_x, bbox[5]+margin_y], [bbox[6]-margin_x, bbox[7]+margin_y]])
    mask = np.ones(ori_img.shape, dtype=np.uint8)
    mask.fill(255)
    cv2.fillPoly(mask, roi_corners, 0)
    img_crop = cv2.bitwise_or(ori_img, mask)
    
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_crop = cv2.warpPerspective(img_crop, M, (w, h))
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    img_crop = (np.reshape(img_crop, (h, w, 1))).transpose(2, 0, 1)
    img_crop = img_crop.astype(np.float32) / 255.
    img_crop = (img_crop - 0.5) / 0.5
    return img_crop

def inference_with_ort(resized_img, ori_img, model_ort_session, sub_ort_session1, sub_ort_session2, nms_thresh, detections_per_img, min_score, secondary_img_size1, secondary_img_size2):
    h0, w0 = ori_img.shape[:2]  # orig hw  
    if secondary_img_size1 != -1:
        (h, w) = secondary_img_size1
    else:
        h, w = 0, 0
    ort_inputs = {model_ort_session.get_inputs()[0].name: resized_img}
    res = model_ort_session.run(None, ort_inputs)
    res = [torch.from_numpy(re) for re in res]
    batched_res = nms([res], detections_per_img, nms_thresh, min_score)
    if sub_ort_session1 == None or sub_ort_session2 == None:
        return 0, 'unknown'

    for res in batched_res:
        preds_str, car_idx = '', 'unknown'
        for bbox, score, label in zip(res['boxes'], res['scores'], res['labels']):
            if score > 0.8 and label == 0:
                bbox = pad_and_reshape(bbox.cpu().numpy(), (1,1), (0,0), (h0, w0))

                img_crop = lpr_pre_process(ori_img, bbox, (h,w))

                ort_inputs = {sub_ort_session1.get_inputs()[0].name: get_batch([img_crop])}
                preds = torch.from_numpy(sub_ort_session1.run(None, ort_inputs)[0]).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * 1)
                preds = preds.permute(1, 0, 2)  # to use CTCloss format
                _, preds_index = preds.max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = decode(preds_index.data, preds_size.data)
                
            if score > 0.8 and label != 0:
                bbox = pad_and_reshape(bbox.cpu().numpy(), (1,1), (0,0), (h0, w0))
                img_crop = ori_img[max(0, int(bbox[1])):int(bbox[5])+1, max(0, int(bbox[0])):int(bbox[4])+1]
                
                try:
                    img_crop = pre_process(img_crop, secondary_img_size2)
                except:
                    print("wait")
                
                ort_inputs = {sub_ort_session2.get_inputs()[0].name: get_batch(img_crop)}
                res = sub_ort_session2.run(None, ort_inputs)[0]
                car_idx = np.argmax(res)
    return car_idx, preds_str

def video_run(video_file_path, primary_img_size, secondary_img_size1, secondary_img_size2, 
                model_ort_session, sub_ort_session1, sub_ort_session2, nms_thresh, detections_per_img, min_score):
    videoCapture = cv2.VideoCapture(video_file_path)
    video_name = video_file_path.strip().split('/')[-1].split('.')[0]
    video_suffix = video_file_path.strip().split('/')[-1].split('.')[1]
    # from n-th frame
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    success, image = videoCapture.read()
    video = cv2.VideoWriter(video_name+'_result.'+video_suffix, cv2.VideoWriter_fourcc(*'XVID'), 25, size)

    cur_num = 0

    while success:
        image_copy = image.copy()
        img_h, img_w, _ = image.shape
        # skip frames
        if cur_num % 1 == 0:
            resized_img = pre_process(image_copy, primary_img_size)
            
            car_idx, preds_str = inference_with_ort(resized_img, image_copy, model_ort_session, sub_ort_session1, sub_ort_session2,
                    nms_thresh, detections_per_img, min_score, secondary_img_size1, secondary_img_size2)
            
            
            if preds_str != '':
                if len(preds_str[0]) < 7 :
                    video.write(image) 
                    success, image = videoCapture.read()
                    cur_num = cur_num + 1
                    continue
                if type(car_idx) == np.int64:
                    print("car label: {}\tlpr: {}".format(car_maker[car_idx], preds_str))
                    image = putText(image, car_maker[car_idx], (img_w-1500, img_h-100),
                                    'NanumSquareB.ttf', (0, 255, 0), int(np.sqrt(img_w/1600*img_h/1200)*100))
                else:
                    print(print("car label: {}\tlpr: {}".format(car_idx, preds_str)))
                    image = putText(image, car_idx, (img_w-1500, img_h-100),
                                    'NanumSquareB.ttf', (0, 255, 0), int(np.sqrt(img_w/1600*img_h/1200)*100))
                image = putText(image, preds_str[0], (img_w-800, img_h-200),
                                    'NanumSquareB.ttf', (0, 0, 255), int(np.sqrt(img_w/1600*img_h/1200)*100))

        
    
        video.write(image) 
        success, image = videoCapture.read()
        cur_num = cur_num + 1
    video.release()
    videoCapture.release()

    return cur_num

@logger.catch
def main(cfg):

    ckpt_file = cfg.ckpt
    model_ort_session = onnxruntime.InferenceSession(ckpt_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    sub_ckpt1 = cfg.get('sub_ckpt1', None)
    sub_ckpt2 = cfg.get('sub_ckpt2', None)
    if sub_ckpt1 is not None:
        sub_ort_session1 = onnxruntime.InferenceSession(sub_ckpt1, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        sub_ort_session1 = None
    if sub_ckpt1 is not None:
        sub_ort_session2 = onnxruntime.InferenceSession(sub_ckpt2, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        sub_ort_session2 = None
    
    logger.info("Run Model..")
    input_path = cfg.data.data_root
    primary_img_size = cfg.data.get('primary_img_size', -1)
    secondary_img_size1 = cfg.data.get('secondary_img_size1', -1)
    secondary_img_size2 = cfg.data.get('secondary_img_size2', -1)
    nms_thresh = cfg.get('nms_thresh', 0.5)
    detections_per_img = cfg.get('detections_per_img', 100)
    min_score = cfg.get('min_score', 0.01)
    
    if cfg.inference_method == 'Image':
        logger.info("Inference with Image..")
        input_list = glob.glob(os.path.join(input_path, '*.jpg'))      
        inference_time = 0
        for path in input_list:
            begin = time.time()
            ori_img = cv2.imread(path)
            resized_img = pre_process(ori_img, primary_img_size)
            car_idx, preds_str = inference_with_ort(resized_img, ori_img, model_ort_session, sub_ort_session1, sub_ort_session2,
                    nms_thresh, detections_per_img, min_score, secondary_img_size1, secondary_img_size2)
            inference_time += time.time() - begin
            if car_idx != 'unknown' and preds_str != '':
                print("car label: {}\tlpr: {}\t{}".format(car_maker[car_idx], preds_str, path.split('/')[-1]))

        print('FPS:', str(int(len(input_list)/(inference_time))))
    
    elif cfg.inference_method == 'Video':
        logger.info("Inference with Video..")
        l_v = glob.glob(os.path.join(input_path,  "*.mp4"))  
        for v in tqdm(l_v):
            begin = time.time()
            cur_num = video_run(v, primary_img_size, secondary_img_size1, secondary_img_size2, 
                model_ort_session, sub_ort_session1, sub_ort_session2, nms_thresh, detections_per_img, min_score)
            end = time.time()
            print('totol_time:', str(end-begin))
            print('totol_frame:', str(cur_num))
            print('FPS:', str(int(cur_num/(end-begin))))
    elif cfg.inference_method == 'Stream':
        i = 0
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 20)
        inference_time = 0
        while True:
            f,ori_img = cap.read()
            img_h, img_w, _ = ori_img.shape
            begin = time.time()
            resized_img = pre_process(ori_img, primary_img_size)
            car_idx, preds_str = inference_with_ort(resized_img, ori_img, model_ort_session, sub_ort_session1, sub_ort_session2,
                    nms_thresh, detections_per_img, min_score, secondary_img_size1, secondary_img_size2)
            inference_time += time.time() - begin
            if preds_str != '':
                if len(preds_str[0]) < 7 :
                    continue
                if type(car_idx) == np.int64:
                    print("car label: {}\tlpr: {}".format(car_maker[car_idx], preds_str))
                    ori_img = putText(ori_img, car_maker[car_idx], (int(img_w*0.1), int(img_h*0.8)),
                                    'NanumSquareB.ttf', (0, 255, 0), int(np.sqrt(img_w/1600*img_h/1200)*100))
                else:
                    print(print("car label: {}\tlpr: {}".format(car_idx, preds_str)))
                    ori_img = putText(ori_img, car_idx, (int(img_w*0.1), int(img_h*0.8)),
                                    'NanumSquareB.ttf', (0, 255, 0), int(np.sqrt(img_w/1600*img_h/1200)*100))
                ori_img = putText(ori_img, preds_str[0], (int(img_w*0.5), int(img_h*0.8)),
                                    'NanumSquareB.ttf', (0, 0, 255), int(np.sqrt(img_w/1600*img_h/1200)*100))
            cv2.imshow("webcam",ori_img)
            if (cv2.waitKey(5) != -1):
                break
            i += 1
        print('FPS:', str(int(i/(end-begin))))
if __name__ == "__main__":

    args = make_parser().parse_args()
    cfg = OrderedDict()
    data = json_to_dict(args.data_cfg)
    cfg.update(data)
    for key in vars(args):
        value = getattr(args, key)
        cfg.update({key: value})
    cfg = AttrDict(cfg)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    main(cfg)

import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import csv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

def parse_opt(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args(args)
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

class CommonDetection:

    def __init__(self, args):
            # write code to change the args dict to command line args
        args_list = []
        for k, v in args.items():
            args[k] = '--' + k
            args_list.append(args[k])
            args_list.append(str(v))
        args = args_list

        ori_dir = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        self.opt = parse_opt(args)
        check_requirements(exclude=('tensorboard', 'thop'))
        # Load model
        self.device = select_device(self.opt.device)
        self.model = DetectMultiBackend(weights=self.opt.weights, device=self.device, dnn=self.opt.dnn, data=self.opt.data, fp16=self.opt.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.opt.imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        os.chdir(ori_dir)
    
    def __call__(self, input_ctx):
        model = self.model
        conf_thres = self.opt.conf_thres
        iou_thres = self.opt.iou_thres
        classes = self.opt.classes
        agnostic_nms = self.opt.agnostic_nms
        max_det = self.opt.max_det

        # Run inference
        im0 = input_ctx['image']
        im = im0.copy()
        im = letterbox(im, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        # with dt[1]:
        pred = model(im)

        # NMS
        # with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print(pred)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # print(f"{n} {self.names[int(c)]}{'s' * (n > 1)}, ")  # add to string
                # Filter the det results, only keep the interested class
                class_index = input_ctx['class_index']
                det = det[det[:, 5] == class_index]
                # Write results
                # draw box and label for each detection
                # for *xyxy, conf, cls in reversed(det):
                #     c = int(cls)  # integer class
                #     label =  f'{self.names[c]} {conf:.2f}'
                #     annotator.box_label(xyxy, label, color=colors(c, True))

                    # cv2.imshow("11", im0)
                    # cv2.waitKey(1)  # 1 millisecond
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # output_ctx = {}
        # output_ctx['image'] = im0
        # 一个包含了所有检测结果的list，每个检测结果包含了一个检测框的坐标、置信度、类别
        # output_ctx['result'] = det
        # print(det.shape[0])

        # only for context analysis data colletction
        return det
    
        # return output_ctx

if __name__ == '__main__':
    args = {
        'weights': 'yolov5s.pt',
        'device': 'cpu',
        'img': 1920,
        # 'device': 'cuda:0'
    }

    detector = CommonDetection(args)
    video_cap = cv2.VideoCapture("/Volumes/Untitled/video/people.mp4")

    while True:
        ret, frame = video_cap.read()

        input_ctx = dict()
        input_ctx['image'] = frame
        input_ctx['class_index'] = 0
        detection_result = detector(input_ctx)
        print('detect one frame')





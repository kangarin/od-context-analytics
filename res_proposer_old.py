import os
from common_detection.common_detection import CommonDetection
# based on the distribution of bbox size, we can tune the resolution 
# while ensuring that the accuracy is under user's constraint.

# res_options = [(1920, 1080), (1600, 900), (1280, 720), (960, 540), (640, 480), (320, 240)]

def calculate_histogram(data, num_bins):
    bin_width = 1 / num_bins  
    frequencies = {i: 0 for i in range(num_bins)}
    for value in data:
        bin_index = min(int(value // bin_width),num_bins-1)
        frequencies[bin_index] += 1
    return frequencies

import numpy as np
def calculate_cumulated_histogram(data, num_bins):
    bin_width = 1 / num_bins  
    frequencies = {i: 0 for i in range(num_bins)}
    for value in data:
        bin_index = min(int(value // bin_width),num_bins-1)
        frequencies[bin_index] += 1
    for i in range(num_bins-2, -1, -1):
        frequencies[i] += frequencies[i+1]
    return frequencies

def relative_size(object_width, object_height, img_width, img_height):
    return object_width / img_width, object_height / img_height

def one_res_distribution(bboxes = [], res = (1920, 1080), num_bins = 20):
    relative_widths = []
    relative_heights = []
    for bbox in bboxes:
        relative_width, relative_height = relative_size(bbox[2] - bbox[0], bbox[3] - bbox[1], res[0], res[1])
        relative_widths.append(relative_width)
        relative_heights.append(relative_height)
    relative_sizes = [max(relative_widths[i], relative_heights[i]) for i in range(len(relative_widths))]
    return calculate_cumulated_histogram(relative_sizes, num_bins)

import csv
def kb_build(profile_data_path, gt_res, num_bins = 20):
    res_profile_dict = {}
    gt_res_profile = None
    for path in profile_data_path:
        res = int(path.split("/")[-1].split(".")[-2])
        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            # skip header
            next(csvreader, None)
            tmp_list = []
            for row in csvreader:
                # only take the xyxy coordinates
                row_revised = [[l[0], l[1], l[2], l[3]] for l in eval(row[1])]
                tmp_list.extend(row_revised)
            # print("res: ", res, ", list size: ", tmp_list.__len__())
            res_profile_dict[res] = tmp_list
            if res == gt_res[0]:
                gt_res_profile = tmp_list

    res_kb = {}
    gt_kb = None
    for i in res_profile_dict.keys():
        if res_profile_dict[i] == gt_res_profile:
            gt_kb = one_res_distribution(res_profile_dict[i], (i, int(i * 9 / 16)), num_bins)
        else:
            res_kb[i] = one_res_distribution(res_profile_dict[i], (i, int(i * 9 / 16)), num_bins)
    # print(res_kb)
    # print(gt_kb)
    return res_kb, gt_kb
                  
def satisfy_constraint(low_res_kb, ori_res_kb, accuracy_constraint):
    return True

# profile data should contains a list of file names, each file
# is a csv file containing the detected bboxes of a certain number of frames of a 
# same video under a certain resolution.
# the name of the file should be in the format of "1920.csv", "1600.csv", etc.
class res_proposer:
    def __init__(self, detector, gt_res, profile_data_path, num_bins = 20, class_index = 0):
        self.gt_res = gt_res
        self.res_kb, self.gt_kb = kb_build(profile_data_path, gt_res, num_bins)
        self.num_bins = num_bins
        self.class_index = class_index
        self.detector = detector
    
    def detect(self, frame):
        input_ctx = dict()
        input_ctx['image'] = frame
        input_ctx['class_index'] = self.class_index
        detection_result = self.detector(input_ctx)
        return [[int(l[0]), int(l[1]), int(l[2]), int(l[3])] for l in detection_result]

    def propose(self, gt_bbox = [], accuracy_constraint = 0.9):
        min_relative_size = self.gt_res[0]
        if gt_bbox == []:
            # return the smallest resolution in res_kb.keys()
            return min(self.res_kb.keys())
            
        for bbox in gt_bbox:
            relative_width, relative_height = relative_size(bbox[2] - bbox[0], bbox[3] - bbox[1], self.gt_res[0], self.gt_res[1])
            min_relative_size = min(min_relative_size, min(relative_width, relative_height))
        min_size_bin_num = int(min_relative_size * 20)
        # find the lowest resolution that is under the accuracy constraint
        lowest_res = self.gt_res[0]
        for res in self.res_kb.keys():
            if self.res_kb[res][min_size_bin_num] != 0 and self.gt_kb[min_size_bin_num] != 0 and self.res_kb[res][min_size_bin_num]/self.gt_kb[min_size_bin_num] >= accuracy_constraint:
            # if satisfy_constraint(self.res_kb[res], self.gt_kb, accuracy_constraint):    
                lowest_res = min(lowest_res, res)
        return lowest_res
    
    def propose_api_test(self, gt_bbox = [], accuracy_constraint = 0.9):
        min_res = 320
        max_res = self.gt_res[0]
        import random
        return random.randint(min_res, max_res)
    


if __name__ == "__main__":
    profile_root_path = "traffic_india_1080_profile/"
    profile_data_path = []
    res_options = []
    gt_res = (0, 0)
    # find all the file in the profile_root_path
    for root, dirs, files in os.walk(profile_root_path):
        for file in files:
            if file.endswith(".csv"):
                profile_data_path.append(os.path.join(root, file))
                cur_res = int(file.split("/")[-1].split(".")[-2])
                res_options.append((cur_res, int(cur_res * 9 / 16)))
                if cur_res > gt_res[0]:
                    gt_res = (cur_res, int(cur_res * 9 / 16))

    args = {
        'weights': 'yolov5s.pt',
        'device': 'cpu',
        'img': gt_res[0],
        # 'device': 'cuda:0'
    }
    detector = CommonDetection(args)
    # 初始化一个proposer，传入离线采集的profile数据作为knowledge base
    p = res_proposer(detector, gt_res, profile_data_path=profile_data_path, num_bins=20, class_index= 2)
    # 传入当前帧的检测结果框，以及精度约束，返回一个建议的分辨率
    # res1 = p.propose(gt_bbox=[[0, 0, 5, 5],[5, 6, 10, 20]], accuracy_constraint=0.4)
    # res2 = p.propose(gt_bbox=[[0, 0, 50, 50],[5, 6, 100, 200]], accuracy_constraint=0.8)
    # res3 = p.propose(gt_bbox=[[0, 0, 5, 5],[5, 6, 10, 20]], accuracy_constraint=0.6)
    # res4 = p.propose(gt_bbox=[[0, 0, 50, 50],[5, 6, 100, 200]], accuracy_constraint=0.9)
    # print(res1)
    # print(res2)
    # print(res3)
    # print(res4)
    import cv2
    video_path = "/Users/wenyidai/Downloads/india-traffic-1080p.mp4"
    cap = cv2.VideoCapture(video_path)
    skip_frame = 500
    while True:
        ret, frame = cap.read()
        if ret:
            skip_frame -= 1
            if skip_frame == 0:
                # 传入当前帧的检测结果框，以及精度约束，返回一个建议的分辨率
                res = p.propose(gt_bbox=p.detect(frame), accuracy_constraint=0.8)
                print(res)
                # cv2.imshow("frame", frame)
                # cv2.waitKey(1)
                skip_frame = 50
        else:
            break
import os
from common_detection.common_detection import CommonDetection
# based on the distribution of bbox size, we can tune the resolution 
# while ensuring that the accuracy is under user's constraint.

def relative_size(object_width, object_height, img_width, img_height):
    return object_width / img_width, object_height / img_height

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
                tmp_list.append(row)
            # print("res: ", res, ", list size: ", tmp_list.__len__())
            res_profile_dict[res] = tmp_list
            if res == gt_res[0]:
                gt_res_profile = tmp_list

    kb = {}
    for i in res_profile_dict.keys():
        # if res_profile_dict[i] == gt_res_profile:
        #     pass
        # else:
        #     kb[i] = calculate_IOU_precision(res_profile_dict[i], gt_res_profile)
        resolution_ratio = i * 1.0 / gt_res[0]
        kb[i] = calculate_IOU_precision(res_profile_dict[i], gt_res_profile, num_bins, resolution_ratio, gt_res)
    return kb
                  
def calculate_IOU_precision(res_profile, gt_profile, num_bins, resolution_ratio, gt_res):
    assert len(res_profile) == len(gt_profile)
    return_correct_dict = {}
    return_mistake_dict = {}
    return_dict = {}
    for i in range(num_bins):
        return_correct_dict[i] = 0
        return_mistake_dict[i]  = 0
        return_dict[i] = 0
    for i in range(len(res_profile)):
        matched_bbox, missed_or_false_bbox = calculate_one_frame_IOU_precision(eval(res_profile[i][1]), eval(gt_profile[i][1]), resolution_ratio, gt_res)
        assign_bbox_to_bins(matched_bbox, return_correct_dict, gt_res)
        assign_bbox_to_bins(missed_or_false_bbox, return_mistake_dict, gt_res)
    # from the last key to the first key to calculate the cumulated precision from right to left
    for i in range(num_bins-2, -1, -1):
        return_correct_dict[i] += return_correct_dict[i+1]
        return_mistake_dict[i] += return_mistake_dict[i+1]
    has_first_nonzero = False
    for i in range(num_bins-1, -1, -1):
        if return_correct_dict[i] == 0 and has_first_nonzero == False:
            return_dict[i] = 1.0
            continue
        if return_correct_dict[i] != 0:
            has_first_nonzero = True
        return_dict[i] = return_correct_dict[i] / (return_correct_dict[i] + return_mistake_dict[i] + 1e-8)
    return return_dict

def assign_bbox_to_bins(bboxes, bin_dict, gt_res):
    bin_num = len(bin_dict.keys())
    bin_width = 1 / bin_num
    for bbox in bboxes:
        relative_width, relative_height = relative_size(bbox[2] - bbox[0], bbox[3] - bbox[1], gt_res[0], gt_res[1])
        s = min(relative_width, relative_height)
        bin_index = min(int(s // bin_width), bin_num-1)
        bin_dict[bin_index] += 1
        

def calculate_one_frame_IOU_precision(one_res_profile, one_gt_profile, resolution_ratio, gt_res):
    from experiment.helper import bbox_iou, bbox_iou_list, visualize_result_bboxes
    # a list of all the size & precision pairs, [[size1, precision1], [size2, precision2], ...]

    one_res_profile = [[l[0] / resolution_ratio, 
                        l[1] / resolution_ratio, 
                        l[2] / resolution_ratio, 
                        l[3] / resolution_ratio] for l in one_res_profile]
    one_gt_profile = [[l[0], l[1], l[2], l[3]] for l in one_gt_profile]
    result = bbox_iou_list(one_res_profile, one_gt_profile, 0.5)
    # three cases: 1. true bbox matches low resolution bbox
    # 2. true bbox does not have a low resolution bbox match
    # 3. low resolution bbox is a false positive
    # case 2 and 3 are combined as one for convenience
    matched_bbox = []
    missed_or_false_bbox = []
    to_remove = []
    for bbox in one_res_profile:
        for r in result:
            if bbox == r[1]:
                matched_bbox.append(bbox)
                to_remove.append(bbox)
                break
    # for bbox in to_remove:
    #     one_res_profile.remove(bbox)
    missed_or_false_bbox = one_res_profile.copy()
    for bbox in to_remove:
        missed_or_false_bbox.remove(bbox)

    for bbox in one_gt_profile:
        is_missed = True
        for r in result:
            if bbox == r[2]:
                is_missed = False
                break
        if is_missed:
            missed_or_false_bbox.append(bbox)
    # if resolution_ratio == 0.25:
    # visualize_result_bboxes(one_res_profile, one_gt_profile, gt_res)
    # visualize_result_bboxes(matched_bbox, matched_bbox, gt_res)
    # for i in one_gt_profile:
    #     for j in one_res_profile:
    #         pass
    return matched_bbox, missed_or_false_bbox
# profile data should contains a list of file names, each file
# is a csv file containing the detected bboxes of a certain number of frames of a 
# same video under a certain resolution.
# the name of the file should be in the format of "1920.csv", "1600.csv", etc.
class res_proposer:
    def __init__(self, detector, gt_res, profile_data_path, num_bins = 20, class_index = 0):
        self.gt_res = gt_res
        self.kb = kb_build(profile_data_path, gt_res, num_bins)
        # use matplotlib to plot self.kb as histograms
        import matplotlib.pyplot as plt
        # for key in self.kb.keys():
        #     plt.clf()
        #     plt.bar(range(len(self.kb[key])), self.kb[key].values(), align='center')
        #     plt.xticks(range(len(self.kb[key])), list(self.kb[key].keys()))
        #     plt.xlabel('relative size')
        #     plt.ylabel('accuracy')
        #     plt.title('accuracy distribution in ' + str(key) + 'p')
        #     plt.show()
        #     plt.savefig('accuracy distribution in ' + str(key) + 'p') 

        # use matplotlib to plot self.kb as line charts

        # for key in self.kb.keys():
        #     plt.clf()
        #     plt.ylim(0.0, 1.1)
        #     # only plot the first 15 keys
        #     plt.plot(list(self.kb[key].keys())[:15], list(self.kb[key].values())[:15])
        #     # plt.plot(list(self.kb[key].keys()), list(self.kb[key].values()))
        #     plt.xlabel('relative size index')
        #     plt.ylabel('accuracy')
        #     plt.title('accuracy distribution in ' + str(key) + 'p')
        #     plt.show()
        #     plt.savefig('accuracy distribution in ' + str(key) + 'p')

        # # take the first kv pair from all kbs of different resolutions
        # first_bin_dict = {}
        # for key in self.kb.keys():
        #     first_bin_dict[key] = self.kb[key][0]
        # # sort from small to large
        # first_bin_dict = dict(sorted(first_bin_dict.items(), key=lambda item: item[0]))
        # # use matplotlib to plot first_bin_dict as line charts
        # plt.clf()
        # plt.ylim(0.0, 1.1)
        # plt.plot(list(first_bin_dict.keys()), list(first_bin_dict.values()))
        # plt.xlabel('resolution')
        # plt.ylabel('accuracy')
        # plt.title('accuracy distribution in the first bin')
        # plt.show()
        # plt.savefig('accuracy distribution in the first bin')



        self.num_bins = num_bins
        self.class_index = class_index
        self.detector = detector
    
    def detect(self, frame):
        input_ctx = dict()
        input_ctx['image'] = frame
        input_ctx['class_index'] = self.class_index
        detection_result = self.detector(input_ctx)
        return [[int(l[0]), int(l[1]), int(l[2]), int(l[3])] for l in detection_result]

    def propose(self, gt_bbox = [], accuracy_constraint = 0.8):
        min_relative_size = self.gt_res[0]
        if gt_bbox == []:
            # return the smallest resolution in kb.keys()
            return min(self.kb.keys())
            
        for bbox in gt_bbox:
            relative_width, relative_height = relative_size(bbox[2] - bbox[0], bbox[3] - bbox[1], self.gt_res[0], self.gt_res[1])
            min_relative_size = min(min_relative_size, min(relative_width, relative_height))
        min_size_bin_num = int(min_relative_size * self.num_bins)
        # find the lowest resolution that is under the accuracy constraint
        lowest_res = self.gt_res[0]
        for res in self.kb.keys():
            if self.kb[res][min_size_bin_num] >= accuracy_constraint:
            # if satisfy_constraint(self.res_kb[res], self.gt_kb, accuracy_constraint):    
                lowest_res = min(lowest_res, res)
        return lowest_res
    
    def propose_api_test(self, gt_bbox = [], accuracy_constraint = 0.9):
        min_res = 320
        max_res = self.gt_res[0]
        import random
        return random.randint(min_res, max_res)
    


if __name__ == "__main__":
    profile_root_path = "traffic_india_profile/"
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
    p = res_proposer(detector, gt_res, profile_data_path=profile_data_path, num_bins=30, class_index = 0)
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
    video_path = "/Volumes/Untitled/video/traffic-india-720p.mp4"
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
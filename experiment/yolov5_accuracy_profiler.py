import csv
from helper import *
# This class takes two csv files as input, each file is the detection result
# of a certain video clip with different resolutions, e.g. 360p and 1080p.
# each line of the file is a detection result of a frame, the format is:
# frame_num, class_id, [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
class yolov5_det_res_input:
    def __init__(self, video_clip_name = "people1.mp4", 
                 low_res_input_path = "", ori_res_input_path = "",
                 low_res = (640, 360), ori_res = (1920, 1080)):
        
        self.video_clip_name = video_clip_name
        self.low_res = low_res
        self.ori_res = ori_res
    
        self.f_low = open(low_res_input_path, 'r')
        self.low_res_reader = csv.reader(self.f_low)
        # remove header
        next(self.low_res_reader)

        self.f_ori = open(ori_res_input_path, 'r')
        self.ori_res_reader = csv.reader(self.f_ori)
        # remove header
        next(self.ori_res_reader)

    def next_frame_det(self):
        try:
            # choose the second column and change to list as return result
            return_res_low = eval(next(self.low_res_reader)[1])
            return_res_ori = eval(next(self.ori_res_reader)[1])
            return return_res_low, return_res_ori
        except StopIteration:
            self.f_low.close()
            self.f_ori.close()
            return None

class yolov5_det_res_output:
    def __init__(self, save_path = ""):
        pass

    def get_output(self):
        pass


from enum import Enum

# class object_size(Enum):
#     XSMALL = 1
#     SMALL = 2
#     MEDIUM = 3
#     LARGE = 4
#     XLARGE = 5

# def object_size_hierarchy_lookup(relative_width, relative_height):
    # relative_size = max(relative_width, relative_height)
    # if relative_size < 0.01:
    #     return object_size.XSMALL
    # elif relative_size < 0.025:
    #     return object_size.SMALL
    # elif relative_size < 0.075:
    #     return object_size.MEDIUM
    # elif relative_size < 0.15:
    #     return object_size.LARGE
    # else:
    #     return object_size.XLARGE
    
def relative_size(object_width, object_height, img_width, img_height):
    return object_width / img_width, object_height / img_height

# class det_size_statistics:
#     def __init__(self):
#         self.xsmall = 0
#         self.small = 0
#         self.medium = 0
#         self.large = 0
#         self.xlarge = 0
    
#     def add(self, size):
#         if size == object_size.XSMALL:
#             self.xsmall += 1
#         elif size == object_size.SMALL:
#             self.small += 1
#         elif size == object_size.MEDIUM:
#             self.medium += 1
#         elif size == object_size.LARGE:
#             self.large += 1
#         elif size == object_size.XLARGE:
#             self.xlarge += 1
    
#     def output_dict(self):
#         return {"XSMALL": self.xsmall, "SMALL": self.small, "MEDIUM": self.medium, "LARGE": self.large, "XLARGE": self.xlarge}

from collections import Counter

def calculate_histogram(data, num_bins):
    # 计算数据范围
    # min_val = min(data)
    # max_val = max(data)
    # bin_width = (max_val - min_val) / num_bins
    bin_width = 1 / num_bins
    
    # 初始化频率字典
    frequencies = {i: 0 for i in range(num_bins)}
    
    # 将数据点分配到相应的bin中
    for value in data:
        # bin_index = min(int((value - min_val) // bin_width),num_bins-1)
        # a non-linear mapping to make the histogram more smooth
        value = - value * (value - 2)
        bin_index = min(int(value // bin_width),num_bins-1)
        frequencies[bin_index] += 1
    print(frequencies)
    return frequencies
# Because objects have different relative sizes (compared to the image size),
# we can divide them into different categories according to their sizes.
# We want to find the relationship between the size of the object and the
# accuracy of the detection result under different resolutions.
# e.g. small objects (0-50px under 1080p, or ~ 2% relative size) may have 
# a detection accuracy of 80% under 1080p, but 50% under 360p.
class yolov5_accuracy_profiler:
    def __init__(self, yolo_det_res_input, yolo_det_res_output, class_id_list = [i for i in range(80)]):
        self.input = yolo_det_res_input
        self.output = yolo_det_res_output
        self.class_id_list = class_id_list
        # self.stats_low = det_size_statistics()
        # self.stats_ori = det_size_statistics()

    def profile(self):
        all_size_list_low = []
        all_size_list_ori = []
        cur = self.input.next_frame_det()
        while cur is not None:
            list1 = [[l[0], l[1], l[2], l[3]] for l in cur[0] if int(l[5]) in self.class_id_list]
            list2 = [[l[0], l[1], l[2], l[3]] for l in cur[1] if int(l[5]) in self.class_id_list]
            for l in list1:
                tmp = relative_size(l[2] - l[0], l[3] - l[1], self.input.low_res[0], self.input.low_res[1])
                all_size_list_low.append(tmp)
                # self.stats_low.add(object_size_hierarchy_lookup(*tmp))
            for l in list2:
                tmp = relative_size(l[2] - l[0], l[3] - l[1], self.input.ori_res[0], self.input.ori_res[1])
                all_size_list_ori.append(tmp)
                # self.stats_ori.add(object_size_hierarchy_lookup(*tmp))
            # res = bbox_iou_list(list1, list2, 0.3)
            # print(res)
            cur = self.input.next_frame_det()
        # print(self.stats_low.output_dict())
        # print(self.stats_ori.output_dict())
        # visualize
        import matplotlib.pyplot as plt
        # plt.scatter([i[0] for i in all_size_list_low], [i[1] for i in all_size_list_low], s=1, c='r', marker='x')
        # plt.scatter([i[0] for i in all_size_list_ori], [i[1] for i in all_size_list_ori], s=1, c='b', marker='o')
        
        # plt.hist([i[0] for i in all_size_list_low], bins=20, density=False, histtype='step', cumulative=False, label='low')
        # plt.hist([i[0] for i in all_size_list_ori], bins=20, density=False, histtype='step', cumulative=False, label='ori')
        # plt.legend(loc='upper left')
        # plt.xlabel('relative size')
        # plt.ylabel('count')
        # plt.title('car bbox distribution under %sp' % self.input.low_res[1])
        # plt.show()

        # calculate the cumulated histogram from right to left
        counts1, bins1, patches1 = plt.hist([-i[0] for i in all_size_list_low], bins=20, density=False, histtype='step', cumulative=True, label='low')
        counts2, bins2, patches2 = plt.hist([-i[0] for i in all_size_list_ori], bins=20, density=False, histtype='step', cumulative=True, label='ori')

        plt.legend(loc='upper left')
        plt.xlabel('relative size')
        plt.ylabel('count')
        plt.title('car bbox distribution under %sp' % self.input.low_res[1])
        plt.show()

        h0 = calculate_histogram([i[0] for i in all_size_list_low], 20)
        h1 = calculate_histogram([i[0] for i in all_size_list_ori], 20)
        # from the last bin to the first bin
        for i in range(19, -1, -1):
            if(h1[i] != 0 and h0[i]/h1[i] < 0.9):
                print(i)
                break

# 在离线阶段，需要得到当前场景在不同分辨率下的profiler数据，分析不同分辨率下物体大小造成的精度影响
# 在初始化阶段，将离线阶段profiler得到的统计分布作为当前proposer的预测模型
# 在运行阶段，根据用户当前的精度约束，以及当前场景的物体大小分布，提出最优的分辨率配置
# TODO：根据tracking的速度信息，提出最优的帧率配置
class yolov5_cfg_proposer:
    def __init__(self):
        self.res = (1920,1080)
        self.fps = 30


    def propose(self, last_frame_bboxes, accuracy_constraint):
        all_size_list = []
        for bbox in last_frame_bboxes:
            tmp = relative_size(bbox[2] - bbox[0], bbox[3] - bbox[1], self.res[0], self.res[1])
            all_size_list.append(tmp)
        h = calculate_histogram([i[0] for i in all_size_list], 20)
        # select the last non-empty bin
        last_bin = 19
        for i in range(19, -1, -1):
            if(h[i] != 0):
                last_bin = i
                break
        



if __name__ == "__main__":
    # test
    # res_list = [(1920, 1080), (1600, 900), (1280, 720), (960, 540), (640, 360), (320, 180)]
    yolo_det_res_input = yolov5_det_res_input(low_res_input_path="person_bbox_profile/1600.csv",
                                            ori_res_input_path="person_bbox_profile/1920.csv", 
                                            low_res=(1600, 900), 
                                            ori_res=(1920, 1080))
    next_frame_det = yolo_det_res_input.next_frame_det()
    yolo_det_res_output = yolov5_det_res_output()
    profiler = yolov5_accuracy_profiler(yolo_det_res_input, yolo_det_res_output)
    profiler.profile()
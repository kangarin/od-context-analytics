import cv2
from common_detection.common_detection import CommonDetection


# based on the distribution of the bbox position of a few continuous frames,
# we can tune the fps while ensuring that when objects disappear from the scene
# a new detection is made (since we cannot forsee the incoming objects).

class tracker_wrapper:
    def __init__(self, tracker, tracker_info):
        self.tracker = tracker
        self.tracker_info = tracker_info

# associate a tracker with a bbox
class tracker_info:
    def __init__(self, init_bbox):
        self.init_bbox = init_bbox
        self.cur_bbox = init_bbox
    
    def update(self, new_bbox):
        self.cur_bbox = new_bbox

def center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

class fps_proposer:
    def __init__(self, detector, gt_res= (1920, 1080), cur_fps = 30, class_index = 0, sliding_window_size = 10, delay_data_path = "delay.csv"):
        self.gt_res = gt_res
        self.cur_fps = cur_fps
        self.detector = detector
        self.class_index = class_index
        self.sliding_window_size = sliding_window_size
        self.sliding_window = []
        import csv
        with open(delay_data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            self.delay_data = list(reader)

    def detect(self, frame):
        input_ctx = dict()
        input_ctx['image'] = frame
        input_ctx['class_index'] = self.class_index
        detection_result = self.detector(input_ctx)
        return [[int(l[0]), int(l[1]), int(l[2]), int(l[3])] for l in detection_result]

    # frames must be continuous
    def propose(self, frame_list = [], init_bbox = []):
        tracker_wrapper_list = []
        for bbox in init_bbox:
            # tracker = cv2.TrackerCSRT_create()
            tracker = cv2.TrackerCSRT_create()
            # change xyxy to xywh
            x1, y1, x2, y2 = bbox
            xywh_bbox = [x1, y1, x2 - x1, y2 - y1]
            tracker.init(frame_list[0], xywh_bbox)
            tmp_tracker_info = tracker_info(bbox)
            tracker_wrapper_list.append(tracker_wrapper(tracker, tmp_tracker_info))

        for i in range(1, len(frame_list)):
            for t in tracker_wrapper_list:
                success, bbox = t.tracker.update(frame_list[i])
                # lose object
                if not success:
                    tracker_wrapper_list.remove(t)
                x, y, w, h = bbox
                bbox_xyxy = [x, y, x + w, y + h]
                t.tracker_info.update(bbox_xyxy)

        # use cv2 to draw a black image, and draw the trajectory on it
        import numpy as np
        img = np.zeros((self.gt_res[1], self.gt_res[0], 3), np.uint8)
        for i in range(len(tracker_wrapper_list)):
            init_x, init_y = center_of_bbox(tracker_wrapper_list[i].tracker_info.init_bbox)
            cur_x, cur_y = center_of_bbox(tracker_wrapper_list[i].tracker_info.cur_bbox) 
            cv2.line(img, (int(init_x), int(init_y)), (int(cur_x), int(cur_y)), (0, 0, 255), 5)
        cv2.imshow("img", img)
        cv2.waitKey(0)

        # initialize the fps to 1
        least_time = 1.0
        # analyze objects' relative moving speed
        for i in range(len(tracker_wrapper_list)):
            init_x, init_y = center_of_bbox(tracker_wrapper_list[i].tracker_info.init_bbox)
            cur_x, cur_y = center_of_bbox(tracker_wrapper_list[i].tracker_info.cur_bbox) 
            x_relative_diff = (cur_x - init_x) / self.gt_res[0]
            y_relative_diff = (cur_y - init_y) / self.gt_res[1]
            time = (1 / self.cur_fps) * (len(frame_list) - 1)
            x_speed = x_relative_diff / time
            y_speed = y_relative_diff / time
            x_predicted_disappear_time = max(0, (self.gt_res[0] - cur_x) / (self.gt_res[0] * (x_speed+1e-5)) if x_speed > 0 else -(cur_x - 0) / (self.gt_res[0] * (x_speed-1e-5)))
            y_predicted_disappear_time = max(0, (self.gt_res[1] - cur_y) / (self.gt_res[1] * (y_speed+1e-5)) if y_speed > 0 else -(cur_y - 0) / (self.gt_res[1] * (y_speed-1e-5)))
            predicted_disappear_time = min(x_predicted_disappear_time, y_predicted_disappear_time)
            least_time= min(least_time, predicted_disappear_time)

        cur_fps = min(1 / (least_time+1e-5), 30)
        self.sliding_window.append(cur_fps)
        if len(self.sliding_window) > self.sliding_window_size:
            self.sliding_window.pop(0)
        return sum(self.sliding_window) / len(self.sliding_window)
    
    def propose_api_test(self, frame_list = [], init_bbox = []):
        assert len(frame_list) > 0
        assert len(init_bbox) > 0
        min_fps = 1
        max_fps = 30
        import random
        return random.randint(min_fps, max_fps)
        

# test
if __name__ == "__main__":
    video_path = "/Volumes/Untitled/video/traffic-india-720p.mp4"

    # # 初始化一个fps_proposer，传入当前的分辨率和fps
    # p = fps_proposer(gt_res= (1920, 1080), cur_fps = 30)
    # # 传入连续的5帧和第一帧的检测结果，对余下4帧进行追踪，并根据追踪结果预测fps
    # p.propose(frame_list = [["first frame"], ["second frame"], ["third frame"], ...],
    #           first_frame_det_bbox = [[40,50,60,70], [10,20,30,40], ...])

    args = {
        'weights': 'yolov5s.pt',
        'device': 'cpu',
        'img': 1280,
        # 'device': 'cuda:0'
    }
    detector = CommonDetection(args)

    video = cv2.VideoCapture(video_path)
    frame_list = []
    fps_history = []
    cnt = 30
    fp = fps_proposer(detector, gt_res= (1280, 720), cur_fps = 30, class_index = 2, sliding_window_size = 10, delay_data_path = "delay.csv")
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_list.append(frame)
        cnt -= 1
        if cnt == 0:
            # detection
            input_ctx = dict()
            input_ctx['image'] = frame_list[0]

            detection_result = fp.detect(frame)
            fps = fp.propose(frame_list, detection_result)
            # use sliding window to get fps
            # fps_history.append(fps)
            # if len(fps_history) > 10:
            #     fps_history = fps_history[1:]
            # fps = sum(fps_history) / len(fps_history)
            print("proposed fps: " + str(fps))

            # write fps to file
            # with open("fps.txt", "a") as f:
            #     f.write(str(fps) + "\n")

            # print("proposed fps: " + str(fps))
            # reset
            cnt = 5
            frame_list = []
    video.release()

        

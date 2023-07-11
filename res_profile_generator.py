# Classes
# names:
#   0: person
#   1: bicycle
#   2: car
#   3: motorcycle
#   4: airplane
#   5: bus
#   6: train
#   7: truck
#   8: boat
#   9: traffic light
#   10: fire hydrant
#   11: stop sign
#   12: parking meter
#   13: bench
#   14: bird
#   15: cat
#   16: dog
#   17: horse
#   18: sheep
#   19: cow
#   20: elephant
#   21: bear
#   22: zebra
#   23: giraffe
#   24: backpack
#   25: umbrella
#   26: handbag
#   27: tie
#   28: suitcase
#   29: frisbee
#   30: skis
#   31: snowboard
#   32: sports ball
#   33: kite
#   34: baseball bat
#   35: baseball glove
#   36: skateboard
#   37: surfboard
#   38: tennis racket
#   39: bottle
#   40: wine glass
#   41: cup
#   42: fork
#   43: knife
#   44: spoon
#   45: bowl
#   46: banana
#   47: apple
#   48: sandwich
#   49: orange
#   50: broccoli
#   51: carrot
#   52: hot dog
#   53: pizza
#   54: donut
#   55: cake
#   56: chair
#   57: couch
#   58: potted plant
#   59: bed
#   60: dining table
#   61: toilet
#   62: tv
#   63: laptop
#   64: mouse
#   65: remote
#   66: keyboard
#   67: cell phone
#   68: microwave
#   69: oven
#   70: toaster
#   71: sink
#   72: refrigerator
#   73: book
#   74: clock
#   75: vase
#   76: scissors
#   77: teddy bear
#   78: hair drier
#   79: toothbrush


import csv
# generate profile data offline, provide the profiled data for res_proposer
from common_detection.common_detection import CommonDetection
import cv2
import os

def generate_one_profile(video_path, weights, res, object_class_index, total_frames, save_root_path):
    args = {
        'weights': weights,
        'img': res[0]
    }
    detector = CommonDetection(args)
    video_cap = cv2.VideoCapture(video_path)
    # if save_root_path does not exist, create it
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    save_file_name = str(res[0]) + ".csv"
    save_file_path = save_root_path + save_file_name
    # create a csv file to store the data
    with open(save_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'number of objects'])
        frame_num = 1
        while True:
            ret, frame = video_cap.read()
            frame = cv2.resize(frame, res)
            if not ret:
                break
            input_ctx = dict()
            input_ctx['image'] = frame
            input_ctx['class_index'] = object_class_index
            detection_result = detector(input_ctx)
            # change tensor to list
            detection_result = detection_result.tolist()
            writer.writerow([frame_num, detection_result])
            print('detected frame ' + str(frame_num))
            frame_num += 1

            if frame_num == total_frames:
                break


if __name__ == '__main__':
    res_options = [
                # (1920, 1080), 
                # (1600, 900), 
                (1280, 720), 
                (960, 540), 
                (640, 480), 
                (320, 240)
                ]
    video_path = "/Volumes/Untitled/video/car-driving.mp4"
    weights = 'yolov5s.pt'
    total_frames = 500
    save_root_path = "car_driving_profile/"

    for res in res_options:
        generate_one_profile(video_path, weights, res, 2, total_frames, save_root_path)
        print("profiled " + str(res) + " finished")
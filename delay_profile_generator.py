# 生成在特定机器用cpu/gpu得到的推理时延和图像大小的关系
import numpy as np
from common_detection.common_detection import CommonDetection

global detector
# 生成与cv2的cap.read()相同格式的图像
def generate_random_img(img_size):
    img = np.random.randint(0, 255, (img_size[1], img_size[0], 3), dtype=np.uint8)
    return img

def generate_delay(device = 'cpu', 
                   img_size = (1920, 1080), 
                   weights = 'yolov5s.pt',
                   repeat = 10):
    args = {
        'weights': weights,
        'device': device,
        'img': img_size[0],
    }
    detector = CommonDetection(args)
    import time
    total_time = 0
    for i in range(repeat):
        img = generate_random_img(img_size)
        input_ctx = dict()
        input_ctx['image'] = img
        start_time = time.time()
        detector(input_ctx)
        end_time = time.time()
        print('time: ', end_time - start_time)
        total_time += end_time - start_time
    print('average time: ', total_time / repeat)
    return total_time / repeat

if __name__ == "__main__":
    resolution_list = [(1920, 1080), (1600, 900), (1280, 720), (960, 540), (640, 480), (320, 240)]
    import csv
    with open("delay.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['resolution', 'delay'])
        for resolution in resolution_list:
            delay = generate_delay(device = 'cpu', 
                                   img_size = resolution, 
                                   weights = 'yolov5s.pt',
                                   repeat = 10)
            writer.writerow([resolution[0], delay])

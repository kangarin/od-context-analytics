import requests
import json
# requests.post('http://localhost:5000/test', json={"name": "John", "age": 30})

import numpy as np
def generate_random_img(img_size):
    img = np.random.randint(0, 255, (img_size[1], img_size[0], 3), dtype=np.uint8)
    return img

    # data = request.get_json()
    # image_type = data['image_type']
    # job_uid = data['job_uid']
    # cam_frame_id = data['cam_frame_id']
    # type = data['type']
    # det_scene = data['det_scene']
    # cur_video_conf = data['cur_video_conf']
    # user_constraint = data['user_constraint']
    # index = -1
    # if type == 'res_profile_frame':
    #     pass
    # elif type == 'fps_profile_frames':
    #     index = data['index']

    # frame_base64 = data['image']

    # 根据上面的信息格式，制造出一个测试json数据，用post请求发送到'http://localhost:5000/test'
    # 用requests.post()方法发送请求，注意json参数的使用

if __name__ == "__main__":
    import cv2
    import base64
    frame = generate_random_img((640, 480))
    _, img_encoded = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(img_encoded).decode('utf-8')

    data = {}
    data['image'] = frame_base64
    data['image_type'] = "jpg"
    data['job_uid'] = "GLOBAL_ID_1"
    data['cam_frame_id'] = "5"
    data['type'] = 'res_profile_frame'
    data['det_scene'] = 'car_detection'
    data['cur_video_conf'] = {"accuracy" : 0.5, "delay" : 30}
    data['user_constraint'] = {"accuracy" : 0.5, "delay" : 30}
    data['index'] = 5
    # print(data)

    res = requests.post('http://localhost:6984/receive_frame', json=data)
    print(res.text)

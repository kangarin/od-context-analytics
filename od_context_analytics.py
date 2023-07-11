from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import threading
import res_proposer
import fps_proposer
from queue import Queue
from queue import deque
app = Flask(__name__)

res_queue = Queue()
fps_queue = Queue()
# 必须与job_manager中保持一致
det_profiler_continous_frames = 5

class od_context_manager:
    def __init__(self, res_queue, fps_queue):
        # 从配置文件中加载可选的场景信息
        self.scene_dict = {}
        # 动态建立当前job_uid代表的场景，根据场景信息定位kb
        self.job_uid_to_scene_dict = {}
        # key: det_scene, value: res_proposer
        self.res_proposers_dict = {}
        # key: det_scene, value: fps_proposer
        self.fps_proposers_dict = {}

        self.res_queue = res_queue
        self.fps_queue = fps_queue
    
    def load_config_and_initialize(self):
        import json
        # 加载config.json配置文件
        with open('config.json', 'r') as f:
            self.scene_dict = json.load(f)
        from common_detection.common_detection import CommonDetection
        for scene in self.scene_dict:
            tmp = self.scene_dict[scene]
            profile_root_path = tmp['profile_root_path']
            class_index = int(tmp['class_index'])
            highest_resolution = int(tmp['highest_resolution'])
            highest_fps = int(tmp['highest_fps'])

            profile_data_path = []
            gt_res = (0, 0)
            # res_options = []
            import os
            for root, dirs, files in os.walk(profile_root_path):
                for file in files:
                    if file.endswith(".csv"):
                        profile_data_path.append(os.path.join(root, file))
                        cur_res = int(file.split("/")[-1].split(".")[-2])
                        # res_options.append((cur_res, int(cur_res * 9 / 16)))
                        if cur_res > gt_res[0]:
                            gt_res = (cur_res, int(cur_res * 9 / 16))

            assert gt_res[0] == highest_resolution
            args = {
                'weights': 'yolov5s.pt',
                'device': 'cpu',
                'img': gt_res[0],
                # 'device': 'cuda:0'
            }
            detector = CommonDetection(args)
            self.res_proposers_dict[scene] = res_proposer.res_proposer(detector = detector,
                                                                       gt_res = gt_res,
                                                                profile_data_path = profile_data_path,
                                                                class_index = class_index)
            self.fps_proposers_dict[scene] = fps_proposer.fps_proposer(detector = detector,
                                                                       gt_res=gt_res,
                                                                cur_fps=highest_fps,
                                                                class_index=class_index)



def process_res_queue(manager, res_queue):
    while True:
        ctx = res_queue.get()
        job_uid = ctx['job_uid']
        det_scene = ctx['det_scene']
        image = ctx['image']
        user_constraint_accuracy = ctx['user_constraint']['accuracy']
        print("get one frame from res_queue!!!!!!!!!!!!")
        # 不在预设的场景中，丢弃
        if det_scene not in manager.scene_dict:
            continue
        # 当前job还没注册过
        if job_uid not in manager.job_uid_to_scene_dict:
            manager.job_uid_to_scene_dict[job_uid] = det_scene
        det = manager.res_proposers_dict[det_scene].detect(image)
        proposed_res = manager.res_proposers_dict[det_scene].propose(det, user_constraint_accuracy)
        print('proposed_res: ' + str(proposed_res))

def process_fps_queue(manager, fps_queue):

    while True:
        ctx = fps_queue.get()
        job_uid = ctx['job_uid']
        det_scene = ctx['det_scene']
        image_list = ctx['image_list']
        user_constraint_delay = ctx['user_constraint']['delay']
        # 不在预设的场景中，丢弃
        if det_scene not in manager.scene_dict:
            continue
        # 当前job还没注册过
        if job_uid not in manager.job_uid_to_scene_dict:
            manager.job_uid_to_scene_dict[job_uid] = det_scene
        det = manager.fps_proposers_dict[det_scene].detect(image_list[0])
        proposed_fps = manager.fps_proposers_dict[det_scene].propose(image_list, det)
        print('proposed_fps: ' + str(proposed_fps))

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/hello/<name>')
def hello(name):
    return render_template('hello.html', name=name)

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    # 解析接收到的请求数据
    data = request.get_json()
    image_type = data['image_type']
    job_uid = data['job_uid']
    cam_frame_id = data['cam_frame_id']
    type = data['type']
    det_scene = data['det_scene']
    cur_video_conf = data['cur_video_conf']
    user_constraint = data['user_constraint']

    q_dict = {
        'image_type': image_type,
        'job_uid': job_uid,
        'cam_frame_id': cam_frame_id,
        'type': type,
        'det_scene': det_scene,
        'cur_video_conf': cur_video_conf,
        'user_constraint': user_constraint
    }

    print('image_type: ' + image_type)
    print('job_uid: ' + job_uid)
    print('cam_frame_id: ' + str(cam_frame_id))
    print('type: ' + type)
    print('det_scene: ' + det_scene)
    print('cur_video_conf: ' + str(cur_video_conf))
    print('user_constraint: ' + str(user_constraint))

    if type == 'res_profile_frame':
        frame_base64 = data['image']
        frame_bytes = base64.b64decode(frame_base64)
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        q_dict['image'] = frame
        res_queue.put(q_dict)
        print("!!!!!!put a res frame!!!!!!")

    elif type == 'fps_profile_frames':
        frame_list = []
        frame_base64_list = data['image_list']
        for frame_base64 in frame_base64_list:
            frame_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_list.append(frame)
        q_dict['image_list'] = frame_list
        fps_queue.put(q_dict)
        print("!!!!!!put fps frames!!!!!!")

    return jsonify({'success': True})



# def process_queue(type, q):
#     global scene_dict
#     global job_uid_to_scene_dict
#     global res_proposers_dict
#     global fps_proposers_dict
#     while True:
#         ctx = q.get()
#         job_uid = ctx['job_uid']
#         det_scene = ctx['det_scene']
#         # 不在预设的场景中，丢弃
#         if det_scene not in scene_dict:
#             continue
#         # 当前job还没注册过
#         if job_uid not in job_uid_to_scene_dict:
#             job_uid_to_scene_dict[job_uid] = det_scene

#         if type == 'res':
#             # TODO
#             proposed_res = res_proposers_dict[det_scene].propose()
#         elif type == 'fps':
#             # TODO
#             proposed_fps = fps_proposers_dict[det_scene].propose()

# def test_queue_valid():
#     while True:
#         tmp = res_queue.get()
#         print("get one frame from res_queue!!!!!!!!!!!!")
#         print("frame_id:" + str(tmp['cam_frame_id']))


if __name__ == '__main__':
    manager = od_context_manager(res_queue, fps_queue)
    manager.load_config_and_initialize()
    processing_process_res = threading.Thread(target=process_res_queue, args=(manager, res_queue))
    processing_process_res.start()
    processing_process_fps = threading.Thread(target=process_fps_queue, args=(manager, fps_queue))
    processing_process_fps.start()

    # p = Process(target=test_queue_valid)
    # p.start()
    app.run(debug=True, host='localhost', port=6984, use_reloader=False)

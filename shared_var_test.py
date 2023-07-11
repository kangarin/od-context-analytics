from flask import Flask, request
from multiprocessing import Process, Queue

app = Flask(__name__)
queue = Queue()

def process_queue():
    while True:
        data = queue.get()
        # 在这里添加你的处理逻辑
        print("Processing data:", data)

@app.route('/test', methods=['POST'])
def handle_post_request():
    data = request.json  # 假设请求数据是JSON格式
    queue.put(data)  # 将请求数据放入队列
    return 'Data received and added to the queue'

if __name__ == '__main__':
    # 创建一个进程来处理队列中的数据
    processing_process = Process(target=process_queue)
    processing_process.start()

    # 启动Flask应用
    app.run()

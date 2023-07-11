import cv2

# 创建一个跟踪器对象
tracker = cv2.TrackerCSRT_create()

# 读取视频文件或摄像头
video = cv2.VideoCapture("/Volumes/Untitled/video/input.mov")  # 如果要从摄像头读取，请将参数设置为 0

# 读取第一帧并选择跟踪目标区域
ret, frame = video.read()
bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

# 初始化跟踪器
tracker.init(frame, bbox)

# 循环处理视频帧
while True:
    # 读取当前帧
    ret, frame = video.read()
    if not ret:
        break

    # 更新跟踪器并获取新的边界框
    success, bbox = tracker.update(frame)

    # 如果成功更新目标跟踪器，则绘制边界框
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示当前帧
    cv2.imshow("Frame", frame)

    # 按下 "q" 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
video.release()
cv2.destroyAllWindows()

# given two bounding boxes, calculate the IoU
def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * \
                    max(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

# calculate the pairwise IoU of two list of bounding boxes,
# choose the pair with the highest IoU and put it into the result list,
# then remove the pair from the two lists and the calculated pairwise IoU,
# repeat the process until one of the list is empty 
# or the highest IoU is lower than the threshold
def bbox_iou_list(box_list1, box_list2, thres):
    ious = []
    result = []
    for i in range(len(box_list1)):
                for j in range(len(box_list2)):
                    iou = bbox_iou(box_list1[i], box_list2[j])
                    if iou > thres:
                        ious.append([iou, box_list1[i], box_list2[j]])
    
    ious.sort(key=lambda x: x[0], reverse=True)
    while len(ious) > 0:
        result.append(ious[0])
        ious.pop(0)
        for i in range(len(ious)):
            if ious[i][1] == result[-1][1] or ious[i][2] == result[-1][2]\
            or ious[i][1] == result[-1][2] or ious[i][2] == result[-1][1]:
                ious.pop(i)
    return result
        

if __name__ == '__main__':
    # generate test cases
    box_list1 = []
    box_list2 = []
    import random
    for i in range(5):
        box_list1.append([random.randint(0, 100), random.randint(0, 100),\
            random.randint(0, 100), random.randint(0, 100)])
        box_list2.append([random.randint(0, 100), random.randint(0, 100),\
            random.randint(0, 100), random.randint(0, 100)])
    print(box_list1)
    print(box_list2)
    # test
    result = bbox_iou_list(box_list1, box_list2, 0.1)
    print(result)

    # visualize the result
    import cv2
    import numpy as np
    img = np.zeros((100, 100, 3), np.uint8)
    for i in range(len(box_list1)):
        cv2.rectangle(img, (box_list1[i][0], box_list1[i][1]),\
            (box_list1[i][2], box_list1[i][3]), (255, 0, 0), 1)
    for i in range(len(box_list2)):
        cv2.rectangle(img, (box_list2[i][0], box_list2[i][1]),\
            (box_list2[i][2], box_list2[i][3]), (0, 0, 255), 1)
    for i in range(len(result)):
        cv2.rectangle(img, (result[i][1][0], result[i][1][1]),\
            (result[i][1][2], result[i][1][3]), (0, 255, 0), 1)
        cv2.rectangle(img, (result[i][2][0], result[i][2][1]),\
            (result[i][2][2], result[i][2][3]), (0, 255, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
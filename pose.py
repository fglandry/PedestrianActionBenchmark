import cv2 as cv
import numpy as np

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


def get_pose_keypoints(img,
                       weights='weights/graph_opt.pb',
                       threshold=0,
                       debug=False):
    net = cv.dnn.readNetFromTensorflow(weights)
    photo_height=img.shape[0]
    photo_width=img.shape[1]
    net.setInput(cv.dnn.blobFromImage(img, 1.0, (photo_width, photo_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))

    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (photo_width * point[0]) / out.shape[3]
        y = (photo_height * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshold else None)

    normalized_keypoints = normalize_keypoints(points, photo_height, photo_width)
    
    if debug:
        for idx, point in enumerate(points):
            cv.ellipse(img, point, (6, 6), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.imshow("cool", img)
            point = [k for k, v in BODY_PARTS.items() if v == idx][0]
            print(point)
            print(normalized_keypoints[idx])
            cv.waitKey(0)

    return normalized_keypoints

def normalize_keypoints(points, photo_height, photo_width):
    norm_points = points.copy()
    for idx, point in enumerate(norm_points):
        try:
            norm_points[idx] = [(point[0] - photo_width/2) / (photo_width/2), \
                        (point[1] - photo_height/2) / (photo_height/2)]
        except:
            norm_points[idx] = [0, 0]
    flat_list_points = [i for sublist in norm_points for i in sublist]
    return flat_list_points

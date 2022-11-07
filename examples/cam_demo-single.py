# zhang

import sys
sys.path.append('../')

import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints
import judge

estimator = BodyPoseEstimator(pretrained=True)

dic = {'focus1':'look at the blackboard', 'focus2':'writing', 'unfocus1':'stretch', 'unfocus2':'cross-legged'}

videoclip = cv2.VideoCapture(0)

while True:
    flag, frame = videoclip.read()
    if not flag:
        break
    keypoints = estimator(frame)

    frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)

    for keypoint in keypoints:
        labels = judge.judge_function(keypoint)
        print(labels)

        left_ears_x = keypoints[0][16][0]
        left_eyes_y = keypoints[0][14][1]

        i = 0
        for x in range(len(labels)):
            if labels[x] == 'None':
                i = i + 1
            else:
                cv2.putText(frame, dic.get(labels[x]), (left_ears_x - 20, left_eyes_y - 20), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), 1)
            if i == 4:
                cv2.putText(frame, 'None', (left_ears_x - 20, left_eyes_y - 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                            (0, 0, 255), 1)

    cv2.imshow('Cam Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoclip.release()
cv2.destroyAllWindows()

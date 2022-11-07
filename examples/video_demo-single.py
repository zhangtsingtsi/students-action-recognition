import os.path
import sys
sys.path.append('../')

import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints
import judge

estimator = BodyPoseEstimator(pretrained=True)

dic = {'focus1':'look at the blackboard', 'focus2':'writing', 'unfocus1':'stretch', 'unfocus2':'cross-legged'}

video_file = 'media/multi.wmv'
videoclip = cv2.VideoCapture(video_file)

width = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = videoclip.get(cv2.CAP_PROP_FPS)

fourcc = int(videoclip.get(cv2.CAP_PROP_FOURCC))

output_file = './output/output_' + (video_file.split("/")[-1:][0])
writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    keypoints = estimator(frame)

    frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)

    labels = judge.judge_function(keypoints)
    print(labels)

    left_ears_x = keypoints[0][16][0]
    left_eyes_y = keypoints[0][14][1]

    i = 0
    for x in range(len(labels)):
        if labels[x] == 'None':
            i = i + 1
        else:
            cv2.putText(frame, dic.get(labels[x]), (left_ears_x-20, left_eyes_y-20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        if i == 4:
            cv2.putText(frame, 'None', (left_ears_x-20, left_eyes_y-20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)

    cv2.imshow('Video Demo', frame)


    key = cv2.waitKey(24)
    writer.write(frame)

    if cv2.waitKey(20) & 0xff == 27: # exit if pressed `ESC`
        break

videoclip.release()
cv2.destroyAllWindows()

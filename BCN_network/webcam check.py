# import cv2
# import numpy as np
#
# input_img_path = "/home/ee303/workspace2019/pix2pix-tensorflow-master/datasets/blister_pack_diffBG/val_old"
#
# img = cv2.imread("{}/1.jpg".format(input_img_path))
# img1 = cv2.imread("{}/2.jpg".format(input_img_path))
# imgA = img[:, 256:512]
# imgB = img1[:, 256:512]
#
# result = np.concatenate((imgA, imgB), axis=1)
# cv2.imwrite("2.jpg", result)

import cv2
import numpy as np

# webcam check

web_num = []
for i in range(8):
    webcam = cv2.VideoCapture(i)
    ret, frame = webcam.read()
    if ret:
        web_num.append(i)
    webcam.release()

print ("web_num = {}".format(web_num))


bottom_webcam = cv2.VideoCapture(web_num[0]) # webcam1 number
top_webcam = cv2.VideoCapture(web_num[1]) # webcam2 number

while (True):
    ret0, frame0 = bottom_webcam.read()
    assert ret0

    ret1, frame1 = top_webcam.read()
    assert ret1

    frame0 = cv2.resize(frame0, (200, 200))
    frame1 = cv2.resize(frame1, (200, 200))

    result = np.concatenate((frame0, frame1), axis=1)

    cv2.imshow("bottom_webcam / top_webcam", result)
    key = cv2.waitKey(1) & 0xFF
    if(key==27):
        bottom_webcam.release()
        top_webcam.release()
        cv2.destroyAllWindows()
        break
import os
import cv2
import glob
import numpy as np

target_path = 'images'
count       = len(os.walk(target_path).__next__()[2])

def getFrame(sec):
    video_cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    ret, image = video_cap.read()

    if ret:
        file_name = "{}/{}_{}.jpg".format(target_path, target_path, count)
        cv2.imwrite(file_name, image) # save frame as JPG file
        print('saved: {}'.format(file_name))
    return ret

videos = glob.glob('videos/*.mp4')
videos = np.array(videos)

for vid in videos:
    print('\n{}'.format(vid))
    video_cap  = cv2.VideoCapture(vid)

    sec       = 0
    frameRate = 0.5 #//it will capture image in each 0.5 second
    # frameRate = 2 #//it will capture image in each 1 second
    success   = getFrame(sec)

    while success:
        count  += 1
        sec     = sec + frameRate
        sec     = round(sec, 2)
        success = getFrame(sec)

    # break
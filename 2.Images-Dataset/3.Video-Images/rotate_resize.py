import cv2
from glob import glob
from tqdm import tqdm
from scipy import ndimage

# target_path = 'rotate_images'
target_path = 'rot_30'

# load the image
images = glob('images/*.jpg')

for i in tqdm(images):
    img    = cv2.imread(i)
    label  = i.split("/")[1]
    nlabel = "{}/rotate_{}".format(target_path, label)

    rotate = ndimage.rotate(img, -90)
    # resize = cv2.resize(rotate, (640, 480))
    resize = cv2.resize(rotate, (300, 300))
    cv2.imwrite(nlabel, resize)

    # cv2.imshow("rotate", rotate)
    # cv2.imshow("resize", resize)
    # cv2.waitKey(0)
    # break
    
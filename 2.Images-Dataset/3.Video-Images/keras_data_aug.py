import cv2
from glob import glob
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator

target_path = 'aug_images'

# load the image
images = glob('images/*.jpg')

# image data generator
image_gen = ImageDataGenerator( rotation_range     = 45, # rotate the image 30 degrees
                                width_shift_range  = 0.1, # Shift the pic width by a max of 10%
                                height_shift_range = 0.1, # Shift the pic height by a max of 10%
                                shear_range        = 0.2, # Shear means cutting away part of the image (max 20%)
                                zoom_range         = 0.3, # Zoom in by 30% max
                                brightness_range   = [0.2,1.0],
                                fill_mode          = 'nearest' # Fill in missing pixels with the nearest filled value
                              )

for i in images:
    img     = cv2.imread(i)
    label   = i.split("/")[1]
    counter = 10

    print("original img: {}".format(label))
    for j in range(counter):
        aug_img   = image_gen.random_transform(img)
        new_label = "{}/{}_{}".format(target_path, j, label)
        print(new_label)
        cv2.imwrite(new_label, aug_img)
    # break
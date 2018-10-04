import cv2
import numpy as np

directory = 'analysis/big/'


def get_one_image(images, name):
    img_list = []
    padding = 500
    for img in images:
        img_list.append(cv2.imread(img))
    max_width = []
    max_height = 0
    for img in img_list:
        max_width.append(img.shape[0])
        max_height += img.shape[1]
    w = np.max(max_width) + padding
    h = max_height + padding

    # create a new array with a size large enough to contain all the images
    final_image = np.ones((h, w, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
        current_y += image.shape[0]
    cv2.imwrite(directory + name, final_image)

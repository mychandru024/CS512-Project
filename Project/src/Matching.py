import cv2
import sys
from matplotlib import pyplot as plt

if __name__ == "__main__":

    image1_name = ""
    image2_name = ""
    if len(sys.argv) < 3:
        cap = cv2.VideoCapture(0)
        if cap.isOpened() != True:
            cap.open()
        ret_val, image = cap.read()
        cv2.imwrite('captured.jpg', image)
        image1_name = "captured.jpg"

        # resizing the captured image
        # comment this line if you want the tranformation to be Rotation
        img_t = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # rotating the captured image anticlockwise by 90 degree
        # comment below 3 lines if you want the tranformation to be scaling
        rows, cols, intensity = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        img_t = cv2.warpAffine(image, M, (cols, rows))

        cv2.imwrite('transformed.jpg', img_t)
        image2_name = "transformed.jpg"
    else:
        image1_name = str(sys.argv[1])
        image2_name = str(sys.argv[2])

    image1 = cv2.imread(image1_name, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_name, cv2.IMREAD_GRAYSCALE)

    print(image1.shape)
    print(image2.shape)

    orb = cv2.ORB_create()

    kp1 = orb.detect(image1, None)
    kp2 = orb.detect(image2, None)

    kp1, des1 = orb.compute(image1, kp1)
    kp2, des2 = orb.compute(image2, kp2)

    img1 = cv2.drawKeypoints(image1, kp1, None, color=(0, 255, 0), flags=0)
    cv2.imwrite('interest_points_in_captured.jpg', img1)
    plt.imshow(img1), plt.show()

    img2 = cv2.drawKeypoints(image2, kp2, None, color=(0, 255, 0), flags=0)
    cv2.imwrite('interest_points_in_transformed.jpg', img2)
    plt.imshow(img2), plt.show()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:30], None, flags=2)
    cv2.imwrite('interest_points_matched_in_captured_and_transformed.jpg', img3)
    plt.imshow(img3), plt.show()






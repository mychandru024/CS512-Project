import cv2


def show(img):
    cv2.imshow('ORB(press esc to exit)',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def opencv_sift(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    out = cv2.drawKeypoints(gray, kp,img)
    return out, len(kp)


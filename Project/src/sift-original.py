import cv2


def show(img):
    cv2.imshow('opencv_sift_function',img)
    cv2.waitKey(0)
    cv2.destroyWindow('opencv_sift_function')

def opencv_sift(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    print(len(kp))
    out = cv2.drawKeypoints(gray, kp,img)
    return out


f = 'data/bolt.jpg'
show(opencv_sift(f))
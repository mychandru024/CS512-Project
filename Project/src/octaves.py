import cv2
import numpy as np
import localization as taylor
import analysis
from scipy.stats import multivariate_normal

def get_btm_dog(d, layer):
    return d[:,:,layer - 1]

def get_top_dog(d, layer):
    return d[:,:,layer + 1]

def detect_extrema(x , y, d, e):
    for i in range(1, 4):
        for x in range(1, x - 1):
            for y in range(1, y - 1):
                arr = np.array([
                    d[x - 1,    y,         i],
                    d[x + 1,    y,         i],
                    d[x - 1,    y - 1,     i],
                    d[x,        y - 1,     i],
                    d[x + 1,    y - 1,     i],
                    d[x - 1,    y + 1,     i],
                    d[x,        y + 1,     i],
                    d[x + 1,    y + 1,     i],
                    d[x - 1,    y,         i - 1],
                    d[x,        y,         i - 1],
                    d[x + 1,    y,         i - 1],
                    d[x - 1,    y - 1,     i - 1],
                    d[x,        y - 1,     i - 1],
                    d[x + 1,    y - 1,     i - 1],
                    d[x - 1,    y + 1,     i - 1],
                    d[x,        y + 1,     i - 1],
                    d[x + 1,    y + 1,     i - 1],
                    d[x - 1,    y,         i + 1],
                    d[x,        y,         i + 1],
                    d[x + 1,    y,         i + 1],
                    d[x - 1,    y - 1,     i + 1],
                    d[x,        y - 1,     i + 1],
                    d[x + 1,    y - 1,     i + 1],
                    d[x - 1,    y + 1,     i + 1],
                    d[x,        y + 1,     i + 1],
                    d[x + 1,    y + 1,     i + 1],
                ])
                lst = np.sort(arr, axis=None)
                if d[x, y, i] < lst[0] or d[x, y, i] > lst[25]:
                    e[x, y, i - 1] = 1;


def find_key_points(image_name, n_octaves, n_scale, threshold, sigma, k):
    img = cv2.imread(image_name)

    # gray scaled original image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filter_sizes = [sigma, sigma * k**1, sigma * k**2, sigma * k**3, sigma * k**4, sigma * k**5]

    # gray scaled image with size doubled
    img_d = cv2.resize(img_gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # gray scaled image with size halved
    img_h = cv2.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # gray scaled image with size quatered
    img_q = cv2.resize(img_h, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # 4 octaves, each octave 6 scale space images
    o1 = np.zeros((img_d.shape[0], img_d.shape[1], 6))
    o2 = np.zeros((img_gray.shape[0], img_gray.shape[1], 6))
    o3 = np.zeros((img_h.shape[0], img_h.shape[1], 6))
    o4 = np.zeros((img_q.shape[0], img_q.shape[1], 6))

    # 5 DoGs in each of the 4 octave
    d1 = np.zeros((img_d.shape[0], img_d.shape[1], 5))
    d2 = np.zeros((img_gray.shape[0], img_gray.shape[1], 5))
    d3 = np.zeros((img_h.shape[0], img_h.shape[1], 5))
    d4 = np.zeros((img_q.shape[0], img_q.shape[1], 5))

    for i in range(0, n_scale):
        o1[:, :, i] = cv2.GaussianBlur(img_d, (5, 5), filter_sizes[i])
        o2[:, :, i] = cv2.GaussianBlur(img_gray, (5, 5), filter_sizes[i])
        o3[:, :, i] = cv2.GaussianBlur(img_h, (5, 5), filter_sizes[i])
        o4[:, :, i] = cv2.GaussianBlur(img_q, (5, 5), filter_sizes[i])

    analysis.save([o1, o2, o3, o4],'__octave__', 6)

    for i in range(0, 5):
        d1[:, :, i] = o1[:, :, i] - o1[:, :, i + 1]
        d2[:, :, i] = o2[:, :, i] - o2[:, :, i + 1]
        d3[:, :, i] = o3[:, :, i] - o3[:, :, i + 1]
        d4[:, :, i] = o4[:, :, i] - o4[:, :, i + 1]

    analysis.print_data([d1, d2, d3, d4], '__dog__', 5)

    # 3 images of extremas in each of the 4 octaves
    e1 = np.zeros((img_d.shape[0], img_d.shape[1], 3))
    e2 = np.zeros((img_gray.shape[0], img_gray.shape[1], 3))
    e3 = np.zeros((img_h.shape[0], img_h.shape[1], 3))
    e4 = np.zeros((img_q.shape[0], img_q.shape[1], 3))

    # finding extrema by comparing the point with its 26 neighbors
    # (8 in same scale space, 9 each in above and below scale space)
    # a pixel should be lower of higher than all the neighbors to be considered an extrema
    for i in range(1, 4):
        for x in range(1, img_d.shape[0] - 1):
            for y in range(1, img_d.shape[1] - 1):
                arr = np.array([
                    d1[x - 1, y, i], d1[x + 1, y, i], d1[x - 1, y - 1, i], d1[x, y - 1, i], d1[x + 1, y - 1, i],
                    d1[x - 1, y + 1, i], d1[x, y + 1, i], d1[x + 1, y + 1, i],
                    d1[x - 1, y, i - 1], d1[x, y, i - 1], d1[x + 1, y, i - 1], d1[x - 1, y - 1, i - 1],
                    d1[x, y - 1, i - 1], d1[x + 1, y - 1, i - 1], d1[x - 1, y + 1, i - 1], d1[x, y + 1, i - 1],
                    d1[x + 1, y + 1, i - 1],
                    d1[x - 1, y, i + 1], d1[x, y, i + 1], d1[x + 1, y, i + 1], d1[x - 1, y - 1, i + 1],
                    d1[x, y - 1, i + 1], d1[x + 1, y - 1, i + 1], d1[x - 1, y + 1, i + 1], d1[x, y + 1, i + 1],
                    d1[x + 1, y + 1, i + 1],
                ])
                lst = np.sort(arr, axis=None)
                if d1[x, y, i] < lst[0] or d1[x, y, i] < lst[25]:
                    e1[x, y, i - 1] = 1;

    for i in range(1, 4):
        for x in range(1, img_gray.shape[0] - 1):
            for y in range(1, img_gray.shape[1] - 1):
                arr = np.array([
                    d2[x - 1, y, i], d2[x + 1, y, i], d2[x - 1, y - 1, i], d2[x, y - 1, i], d2[x + 1, y - 1, i],
                    d2[x - 1, y + 1, i], d2[x, y + 1, i], d2[x + 1, y + 1, i],
                    d2[x - 1, y, i - 1], d2[x, y, i - 1], d2[x + 1, y, i - 1], d2[x - 1, y - 1, i - 1],
                    d2[x, y - 1, i - 1], d2[x + 1, y - 1, i - 1], d2[x - 1, y + 1, i - 1], d2[x, y + 1, i - 1],
                    d2[x + 1, y + 1, i - 1],
                    d2[x - 1, y, i + 1], d2[x, y, i + 1], d2[x + 1, y, i + 1], d2[x - 1, y - 1, i + 1],
                    d2[x, y - 1, i + 1], d2[x + 1, y - 1, i + 1], d2[x - 1, y + 1, i + 1], d2[x, y + 1, i + 1],
                    d2[x + 1, y + 1, i + 1],
                ])
                lst = np.sort(arr, axis=None)
                if d2[x, y, i] < lst[0] or d2[x, y, i] < lst[25]:
                    e2[x, y, i - 1] = 1;

    for i in range(1, 4):
        for x in range(1, img_h.shape[0] - 1):
            for y in range(1, img_h.shape[1] - 1):
                arr = np.array([
                    d3[x - 1, y, i], d3[x + 1, y, i], d3[x - 1, y - 1, i], d3[x, y - 1, i], d3[x + 1, y - 1, i],
                    d3[x - 1, y + 1, i], d3[x, y + 1, i], d3[x + 1, y + 1, i],
                    d3[x - 1, y, i - 1], d3[x, y, i - 1], d3[x + 1, y, i - 1], d3[x - 1, y - 1, i - 1],
                    d3[x, y - 1, i - 1], d3[x + 1, y - 1, i - 1], d3[x - 1, y + 1, i - 1], d3[x, y + 1, i - 1],
                    d3[x + 1, y + 1, i - 1],
                    d3[x - 1, y, i + 1], d3[x, y, i + 1], d3[x + 1, y, i + 1], d3[x - 1, y - 1, i + 1],
                    d3[x, y - 1, i + 1], d3[x + 1, y - 1, i + 1], d3[x - 1, y + 1, i + 1], d3[x, y + 1, i + 1],
                    d3[x + 1, y + 1, i + 1],
                ])
                lst = np.sort(arr, axis=None)
                if d3[x, y, i] < lst[0] or d3[x, y, i] < lst[25]:
                    e3[x, y, i - 1] = 1;

    for i in range(1, 4):
        for x in range(1, img_q.shape[0] - 1):
            for y in range(1, img_q.shape[1] - 1):
                arr = np.array([
                    d4[x - 1, y, i], d4[x + 1, y, i], d4[x - 1, y - 1, i], d4[x, y - 1, i], d4[x + 1, y - 1, i],
                    d4[x - 1, y + 1, i], d4[x, y + 1, i], d4[x + 1, y + 1, i],
                    d4[x - 1, y, i - 1], d4[x, y, i - 1], d4[x + 1, y, i - 1], d4[x - 1, y - 1, i - 1],
                    d4[x, y - 1, i - 1], d4[x + 1, y - 1, i - 1], d4[x - 1, y + 1, i - 1], d4[x, y + 1, i - 1],
                    d4[x + 1, y + 1, i - 1],
                    d4[x - 1, y, i + 1], d4[x, y, i + 1], d4[x + 1, y, i + 1], d4[x - 1, y - 1, i + 1],
                    d4[x, y - 1, i + 1], d4[x + 1, y - 1, i + 1], d4[x - 1, y + 1, i + 1], d4[x, y + 1, i + 1],
                    d4[x + 1, y + 1, i + 1],
                ])
                lst = np.sort(arr, axis=None)
                if d4[x, y, i] < lst[0] or d4[x, y, i] < lst[25]:
                    e4[x, y, i - 1] = 1;

    number_of_extremas = np.count_nonzero(e1) + np.count_nonzero(e2) + np.count_nonzero(e3) + np.count_nonzero(e4);

    """
    #analysis.print_data([e1, e2, e3, e4], '__extreme__', 3)

    (b1, b2, b3) = get_btm_dog(d1, 2), get_btm_dog(d2, 3), get_btm_dog(d3, 4)
    (t1, t2, t3) = get_top_dog(d1, 2), get_top_dog(d2, 3), get_top_dog(d3, 4)


    taylor.find_extrema(e1, b1, d1[:, :, 2], t1, 0)
    taylor.find_extrema(e1, b1, d1[:, :, 3], t1, 1)
    taylor.find_extrema(e1, b1, d1[:, :, 4], t1, 2)
    """

    # magnitudes of the local images in all scale space
    m1 = np.zeros((img_d.shape[0], img_d.shape[1], 3))
    m2 = np.zeros((img_gray.shape[0], img_gray.shape[1], 3))
    m3 = np.zeros((img_h.shape[0], img_h.shape[1], 3))
    m4 = np.zeros((img_q.shape[0], img_q.shape[1], 3))

    # gradient orientation of the keypoints in local images in all scale spaces
    or1 = np.zeros((img_d.shape[0], img_d.shape[1], 3))
    or2 = np.zeros((img_gray.shape[0], img_gray.shape[1], 3))
    or3 = np.zeros((img_h.shape[0], img_h.shape[1], 3))
    or4 = np.zeros((img_q.shape[0], img_q.shape[1], 3))

    for i in range(0, 3):
        for x in range(1, img_d.shape[0] - 1):
            for y in range(1, img_d.shape[1] - 1):
                m1[x, y, i] = np.sqrt(((img_d[x + 1, y] - img_d[x - 1, y]) ** 2) + ((img_d[x, y + 1] - img_d[x, y - 1]) ** 2))
                or1[x, y, i] = np.rad2deg(np.arctan((img_d[x, y + 1] - img_d[x, y - 1]) / (img_d[x + 1, y] - img_d[x - 1, y])))

    for i in range(0, 3):
        for x in range(1, img_gray.shape[0] - 1):
            for y in range(1, img_gray.shape[1] - 1):
                m2[x, y, i] = np.sqrt(((img_gray[x + 1, y] - img_gray[x - 1, y]) ** 2) + ((img_gray[x, y + 1] - img_gray[x, y - 1]) ** 2))
                or2[x, y, i] = np.rad2deg(np.arctan((img_gray[x, y + 1] - img_gray[x, y - 1]) / (img_gray[x + 1, y] - img_gray[x - 1, y])))

    for i in range(0, 3):
        for x in range(1, img_h.shape[0] - 1):
            for y in range(1, img_h.shape[1] - 1):
                m3[x, y, i] = np.sqrt(((img_h[x + 1, y] - img_h[x - 1, y]) ** 2) + ((img_h[x, y + 1] - img_h[x, y - 1]) ** 2))
                or3[x, y, i] = np.rad2deg(np.arctan((img_h[x, y + 1] - img_h[x, y - 1]) / (img_h[x + 1, y] - img_h[x - 1, y])))

    for i in range(0, 3):
        for x in range(1, img_q.shape[0] - 1):
            for y in range(1, img_q.shape[1] - 1):
                m4[x, y, i] = np.sqrt(((img_q[x + 1, y] - img_q[x - 1, y]) ** 2) + ((img_q[x, y + 1] - img_q[x, y - 1]) ** 2))
                or4[x, y, i] = np.rad2deg(np.arctan((img_q[x, y + 1] - img_q[x, y - 1]) / (img_q[x + 1, y] - img_q[x - 1, y])))

    #print(or1)
    keypoints = np.zeros((int(number_of_extremas), 4))
    orient_hist = np.zeros([36, 1])
    count = 0
    for i in range(0, 3):
        for x in range(1, img_d.shape[0] - 1):
            for y in range(1, img_d.shape[1] - 1):
                if e1[x, y, i]:
                    gaussian_window = multivariate_normal(mean=[x, y], cov=((1.5 * filter_sizes[i]) ** 2))
                    bin_index = np.clip(np.floor(or1[x, y, i]), 0, 35)
                    orient_hist[int(np.floor(bin_index))] = m1[x, y, i] * gaussian_window.pdf([x, y])
                    #print(gaussian_window)

                    maxval = np.amax(orient_hist)
                    maxidx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), filter_sizes[i], maxidx])
                    count += 1
                    orient_hist[maxidx] = 0
                    newmaxval = np.amax(orient_hist)
                    while newmaxval > 0.8 * maxval:
                        newmaxidx = np.argmax(orient_hist)
                        np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), filter_sizes[i], newmaxidx]]), axis=0)
                        orient_hist[newmaxidx] = 0
                        newmaxval = np.amax(orient_hist)

find_key_points("data/dp.jpg", 4, 6, 0.03, 1.6, 1.3)
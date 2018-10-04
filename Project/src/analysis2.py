import cv2
import numpy as np
import concat

filename = '_image'
save_dir = 'analysis/'
out_dir = 'output/'

gen = False


def save(images, type, total):
    if not gen:
            return
    ims = []
    for each in range(4):
        for i in range(total):
            file_name = save_dir + type + str(each + 1) + '__' + str(i + 1) + filename + '.jpg'
            ims.append(file_name)
            cv2.imwrite(file_name, images[each][i, :, :])
    concat.get_one_image(ims, type+str('.jpg'))


def print_data(images, name, total):
    if not gen:
            return
    ims = []
    for each in range(4):
        for i in range(total):
            f = save_dir + str(each + 1) + '_s_' + str(i + 1) + name + filename + '.png'
            ims.append(f)
            img = np.array(images[each][i, :, :], dtype=np.uint8)
            cv2.imwrite(f, img)
    concat.get_one_image(ims, name + str('.jpg'))


def print_extrema(images, name, total):
    if not gen:
        return
    ims = []
    for each in range(4):
        for i in range(total):
            f = save_dir + str(1) + '_s_' + str(each) + str(i + 1) + name + filename + '.png'
            img = np.array(images[each][i, :, :] * 255, dtype=np.uint8)
            ims.append(f)
            cv2.imwrite(f, img)
    concat.get_one_image(ims, name + str('.jpg'))


def print_logs(msg):
    print(msg)


def plot_arrows(image, mag, grad, ext):
    s = image.shape
    for r in range(s[0]):
        for c in range(s[1]):
            if ext[r][c] == 0:
                continue
            theta = grad[r][c]
            m = np.abs(mag[r][c])/10
            x2 = int(r + (m * 10) * np.cos(theta))
            y2 = int(c + (m * 10) * np.sin(theta))
            cv2.arrowedLine(image, (c, r), (y2, x2), (0, 255, 100), 1)
    f = out_dir + 'plot' + str(np.random.randint(1000)) + '.png'
    cv2.imwrite(f, image)
    cv2.imshow(f, image)
    cv2.waitKey(0)
    cv2.destroyWindow(f)


def plot(image, mag, grad, ext):
    s = image.shape
    for r in range(s[0]):
        for c in range(s[1]):
            if ext[r][c] == 0:
                continue
            cv2.arrowedLine(image, (c, r), (c, r), (0, 255, 100), 2)
    f = out_dir + 'plot' + str(np.random.randint(1000)) + '.png'
    cv2.imwrite(f, image)
    cv2.imshow(f, image)
    cv2.waitKey(0)
    cv2.destroyWindow(f)

def plot_orientations(image, mag, grad):

    s = image.shape
    D = np.random.randint(s[0]-1, size=s[0])
    DD = np.random.randint(s[0]-1, size=s[1])

    count = 0
    for r in range(s[0]-100):
        for c in range(s[1]-100):
            if mag[r][c] == 0:
                continue
            theta = grad[r][c]
            m = np.abs(mag[r][c])/10
            x2 = int(r + (m+10) * np.cos(theta))
            y2 = int(c + (m+10) * np.sin(theta))
            count += 1
            cv2.arrowedLine(image, (x2, y2), (D[count], DD[count]), (0, 255, 100), 1)
            if count == 100:
                print("press esc to exit this view")
                cv2.imshow('test', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                exit(0)


def get_8_bin_histogram_pos(w):
    if w <= 45:
        return 0
    if w <= 90:
        return 1
    if w <= 135:
        return 2
    if w <= 180:
        return 3
    if w <= 225:
        return 4
    if w <= 270:
        return 5
    if w <= 315:
        return 6
    if w <= 360:
        return 7


def get_bin_pos(value):
    return int((value + 10) / 10) * 10

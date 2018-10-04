import numpy as np


def trace(H):
    """
    Trace :
    The trace (often abbreviated to "tr") is
    related to the derivative of the determinant.
    :param H:
    :return: it return the trace
    """
    return np.power(H['dxx'] + H['dyy'], 2.)


def det(H):
    """
    Det: computes the det of matrix
    :param H:
    :return: deteminant is a float value
    """
    return H['dxx'] * H['dyy'] - np.power(H['dxy'], 2.)


def check_principle_curvature(H):
    """
    This calculates the principle curvature check
    please refer paper D.G Lowe refer section 4.1
    :param H:
    :return:
    """
    r = 10.0
    R = np.power(r + 1, 2.) / r
    if det(H) == 0: return

    ratio = trace(H) / det(H)
    return ratio < R


def is_closer_to_different_point(s):
    """
    This  means the extrema is closer to different sample point.
    :param s:
    :return:
    """
    if np.abs(s[0]) < 0.5 and np.abs(s[1]) < 0.5:
        return True


def extrema_check(D):
    return np.abs(D) > 0.03


def key_point_check(D, H, sol):

    return is_closer_to_different_point(sol) and \
        check_principle_curvature(H) and \
        extrema_check(D)


def get_hessian_dict(h):
    '''
    creating hessian dictonary just for better code readability
    :param h: 
    :return: 
    '''
    return {'dxx':h[0, 0],
            'dxy':h[0, 1],
            'dxs':h[0, 2],
            'dyx':h[1, 0],
            'dyy':h[1, 2],
            'dys':h[1, 2],
            'dsx':h[2, 0],
            'dsy':h[2, 2],
            'dss':h[2, 2]}


def find_extrema(local_extrema, dog_bottom, dog, dog_top):
    """
    Refer the Acurate keypoint localization section from the paper.
    :param local_extrema: This is a matrix of size equal to DOG
    :param dog_bottom: DOG bottom, is used to compute the derivative across sigma
    :param dog_top: DOG top, is used to compute the derivative across sigma
    :param dog: in between dog
    :return E: returns the extream points
    """

    s = local_extrema.shape
    E = np.zeros((s[0], s[1]))
    for r in range(s[0]):
        for c in range(s[1]):
            if local_extrema[r][c] == 1:
                dx = (dog[r,      c + 1,] - dog[r,     c - 1]) / 2.0
                dy = (dog[r + 1,      c ] - dog[r - 1,     c]) / 2.0

                ds = (dog_top[r, c] - dog_bottom[r,  c]) / 2.0

                dxx = (dog[r,  c + 1] + dog[r, c - 1] - 2 * dog[r, c]) * 1.0 / 255
                dyy = (dog[r + 1, c ] + dog[r + 1, c] - 2 * dog[r, c]) * 1.0 / 255

                dxy = (dog[r + 1, c + 1] +
                       dog[r + 1, c - 1] -
                       dog[r - 1, c + 1] -
                       dog[r - 1, c - 1]) * 0.25 / 255

                dxs = (dog[r + 1, c + 1] -
                       dog[r - 1, c + 1] -
                       dog[r + 1, c - 1] +
                       dog[r - 1, c - 1]) * 0.25 / 255

                dys = (dog_top[r + 1, c] -
                       dog_top[r - 1, c] -
                       dog_bottom[r + 1, c] +
                       dog_bottom[r - 1, c]) * 0.25 / 255

                dss = (dog_top[r, c] + dog_bottom[r, c] - 2 * dog[r, c]) * 1.0 / 255

                D = np.matrix([[dx], [dy], [ds]])
                H = np.matrix([[dxx, dxy, dxs],
                               [dxy, dyy, dys],
                               [dxs, dys, dss]])
                hessian = get_hessian_dict(H)
                sol = np.linalg.lstsq(H, D)[0]
                d_x_hat = dog[r, c] + 0.5 * np.dot(D.transpose(), sol)

                if key_point_check(d_x_hat, hessian, sol):
                    E[r][c] = 1
    return E


def magnitude(p1, p2, p3, p4):
    return np.sqrt(np.power(int(p1) - int(p2), 2.) + np.power(int(p3)-int(p4), 2.))


def orientation(p1, p2, p3, p4):
    # arctan2 of the angle formed by(x, y) and the positive x - axis.
    # So a radian is about 360 /(2 * pi)
    # here we are using 36 bins of size 10, so 360/(10*2*np.pi)
    return np.arctan2((int(p1)-int(p2)), (int(p3)-int(p4)))*(360 / (20 * np.pi))


def cal_ori_mag(img):
    s = img.shape
    mag = np.zeros((s[0], s[1]))
    ori = np.zeros((s[0], s[1]))
    for r in range(1, s[0]-1):
        for c in range(1, s[1]-1):
            mag[r][c] = magnitude(img[r+1][c], img[r-1][c], img[r][c+1], img[r][c-1])
            ori[r][c] = orientation(img[r+1][c], img[r-1][c], img[r][c+1], img[r][c-1])
    return ori, mag





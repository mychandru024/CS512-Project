import cv2
import numpy as np
from scipy.stats import multivariate_normal
import localization as taylor
import analysis2 as analysis
import orb

sigma = 1.6
threshold = 0.03
k = 1.3
WINDOW_SIZE = 16
BINS = np.arange(36) + 1
BINS_8 = np.matrix([0, 45, 90, 135, 180, 225, 270, 315, 360])
DEFAULT = np.zeros((16, 16))

sigmas = np.array([1.6, 1.6 * k, 1.6 * (k ** 2),
                   1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5),
                   1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8),
                   1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11)])


class SIFT:
    """
        SIFT
    """
    total_count = 0
    n_octaves = 4
    n_scale = 6

    octave = []
    dog = []
    extrema = []
    mag = []
    grad = []
    o_histogram = []

    octave_size = {}

    def __init__(self, image_path):

        self.img = cv2.imread(image_path)
        self.f = image_path

        # gray scaled original image
        self.img_g = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.octave_size["o2"] = self.img_g.shape

        self.filter_sizes = [sigma,
                             sigma * k ** 1,
                             sigma * k ** 2,
                             sigma * k ** 3,
                             sigma * k ** 4,
                             sigma * k ** 5]

        # gray scaled image with size doubled
        self.img_d = cv2.resize(self.img_g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        self.octave_size["o1"] = self.img_d.shape

        # gray scaled image with size halved
        self.img_h = cv2.resize(self.img_g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        self.octave_size["o3"] = self.img_h.shape

        # gray scaled image with size quatered
        self.img_q = cv2.resize(self.img_h, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        self.octave_size["o4"] = self.img_q.shape

        print("Initial Set of images computing")
        self.compute()

    def init_mat_zeros(self, dim_z):
        temp = []
        for i in range(4):
            shape = self.octave_size["o" + str(i + 1)]
            temp.append(np.zeros((dim_z, shape[0], shape[1])))
        return temp

    def init_octaves(self):
        self.octave = self.init_mat_zeros(6)

    def get_image(self, oct_id):
        if oct_id == 1:
            return self.img_d
        if oct_id == 2:
            return self.img_g
        if oct_id == 3:
            return self.img_h
        if oct_id == 4:
            return self.img_q

    def init_dogs(self):
        self.dog = self.init_mat_zeros(5)

    def init_extrema(self):
        self.extrema = self.init_mat_zeros(3)

    def convolve_gauss(self):
        for each in range(4):
            for i in range(0, self.n_scale):
                self.octave[each][i, :, :] = cv2.GaussianBlur(self.get_image(each + 1), (5, 5), self.filter_sizes[i])

    def compute_dogs(self):
        for each in range(4):
            for i in range(0, 5):
                self.dog[each][i, :, :] = self.octave[each][i, :, :] - self.octave[each][i + 1, :, :]

    def init_grad(self):
        self.grad = self.extrema = self.init_mat_zeros(3)

    def init_mag(self):
        self.mag = self.extrema = self.init_mat_zeros(3)

    def get_image_shape(self, octave):
        return self.octave_size["o" + str(octave + 1)]

    def detect_interest_points(self):
        for octave in range(len(self.extrema)):
            for i in range(1, 4):
                for x in range(1, self.get_image_shape(octave)[0] - 1):
                    for y in range(1, self.get_image_shape(octave)[1] - 1):
                        arr = np.array([
                            self.dog[octave][i, x - 1, y],
                            self.dog[octave][i, x + 1, y],
                            self.dog[octave][i, x - 1, y - 1],
                            self.dog[octave][i, x, y - 1],
                            self.dog[octave][i, x + 1, y - 1],
                            self.dog[octave][i, x - 1, y + 1],
                            self.dog[octave][i, x, y + 1],
                            self.dog[octave][i, x + 1, y + 1],
                            self.dog[octave][i - 1, x - 1, y],
                            self.dog[octave][i - 1, x, y],
                            self.dog[octave][i - 1, x + 1, y],
                            self.dog[octave][i - 1, x - 1, y - 1],
                            self.dog[octave][i - 1, x, y - 1],
                            self.dog[octave][i - 1, x + 1, y - 1],
                            self.dog[octave][i - 1, x - 1, y + 1],
                            self.dog[octave][i - 1, x, y + 1],
                            self.dog[octave][i - 1, x + 1, y + 1],
                            self.dog[octave][i + 1, x - 1, y],
                            self.dog[octave][i + 1, x, y],
                            self.dog[octave][i + 1, x + 1, y],
                            self.dog[octave][i + 1, x - 1, y - 1],
                            self.dog[octave][i + 1, x, y - 1],
                            self.dog[octave][i + 1, x + 1, y - 1],
                            self.dog[octave][i + 1, x - 1, y + 1],
                            self.dog[octave][i + 1, x, y + 1],
                            self.dog[octave][i + 1, x + 1, y + 1],
                        ])
                        lst = np.sort(arr, axis=None)
                        if self.dog[octave][i, x, y] < lst[0] or self.dog[octave][i, x, y] > lst[25]:
                            self.extrema[octave][i - 1, x, y] = 1

    def apply_taylor(self):
        for octave in range(len(self.extrema)):
            for i in range(1, 3):
                self.extrema[octave][i, :, :] = \
                    taylor.find_extrema(self.extrema[octave][i, :, :],
                                        self.dog[octave][i - 1, :, :],  # bottom
                                        self.dog[octave][i, :, :],  # middle
                                        self.dog[octave][i + 1, :, :])  # top

    def stats(self):
        for octave in range(len(self.extrema)):
            for p in range(3):
                s = np.sum(self.extrema[octave][p])
                self.total_count += s
                print("Extrema {} {} found {} interest points".format(octave, p, s))
        print("Total Features detected ", self.total_count)

    def plot_orientations(self):
        for i in range(4):
            analysis.plot_arrows(self.get_image(i + 1),
                                 self.grad[i],
                                 self.mag[i],
                                 self.extrema[i][0, :, :])

    def plot(self):
        for i in range(4):
            analysis.plot(self.get_image(i + 1),
                          self.grad[i],
                          self.mag[i],
                          self.extrema[i][0, :, :])

    def comput_grad_mag(self):
        for i in range(4):
            out = taylor.cal_ori_mag(self.get_image(i + 1))
            self.grad.append(out[0])
            self.mag.append(out[1])

    @staticmethod
    def get_gauss_window(size, _sigma):
        x = np.linspace(0, size, size, endpoint=False)
        y = multivariate_normal.pdf(x, mean=8, cov=1.5 * _sigma)
        y = y.reshape(1, 16)
        return np.dot(y.T, y)

    @staticmethod
    def sub_region(region, r, c):
        """
        It returns a 16x16 window to user
        :param region: Interested features
        :param r: row
        :param c: column
        :return:
        """
        try:
            return region[r - 8:r + 8, c - 8:c + 8]
        except IndexError:
            # print("KNOWN ISSUE :: Corner case is not handled")  # todo Handle edge cases.
            return DEFAULT

    @staticmethod
    def compute_orientation_histogram(weights, orientation, x, y):
        """
        We need to calculate the orientation histogram
        by dividing into 4 parts.
        So consider below, now we need to calculate
        the histogram for each part.
        Basic steps:
        1. Part by part create histograms
            Example:
            suppose orientation is [ 20, 30, 40, 55, 52, 43, 0, 0, degree] in part 1.
            consider the bin size as 90 degree.

                 | /
            _____|/_____
                 |
                 |
            histogram will look like above for bin size = 90.
            Note: In this program we used bin-size = 36

         ___________________________________
        |___|____|____|___+____|___|____|___|
        |___|____|____|___+____|___|____|___|
        |___|____1____|___+____|___2____|___|
        |___|____|____|___+____|___|____|___|
        |++++++++++++++++++++++++++++++++++++
        |___|____|____|___+____|___|____|___|
        |___|____3____|___+____|___4____|___|
        |___|____|____|___+____|___|____|___|
        |___|____|____|___+____|___|____|___|

        above can be sliced into 4 parts like below

        More: Refer paper(D.G Lowe) section 6.

        """

        '''
        Add weights to bins from weight matrix
        
        Orientation angles(in degrees)
             __________
            |30|33 | 45| 
            | .| . | . | 
        O = | .|55 |22 | 
            |4 |_._|_._| 
        
        The histogram for above orientation look like below
        using two bins of size 30 degree
        
        Step 1: Create histogram 
        
                |
                |
                |    
            4   |    2
        --------|---------
          0-30     0-60  
          
        Step 2: Add weights to above histogram
        
        Weights: add this to above histogram
        
                _________  
              |1 |3  | 1 |       
              | .| . | . |      
         W =  | .|1  |2  |       
              |1.|_._|_._|
            
        Algorithm:
           1. Scan the Orientation matrix(O).
                for each angle in O, add the weight to corresponding bin.
                
                Example:
                    O[0, 0] = 30, so add the weight 1 to bin 0 - 30.
                    O[0, 1] = 33, so add the weight 3 to bin 0 - 30.
                    O[1, 2] = 22, so add the weight 2 to bin 0 - 30.
                    O[3, 1] = 4 , so add the weight 1 to bin 0 - 30.
                    
                    Total weights for 0-30 bin = (1 + 3 + 2 + 1 ) = 7
                    
                HOG is below. Histogram which takes 
                weights and gradient directions.
                
                
                    ^   |
                    ^   |
                    ^   |
                    ^   |
                    ^   |    
                    ^   |    ^
                    ^   |    ^
                --------|---------
                  0-30    0-60  
        
        '''

        part1 = np.ravel(np.abs(orientation[0:8, 0:8]))
        part2 = np.ravel(np.abs(orientation[0:8, 8:16]))
        part3 = np.ravel(np.abs(orientation[8:16, 0:8]))
        part4 = np.ravel(np.abs(orientation[8:16, 8:16]))
        pieces = [part1, part2, part3, part4]

        part_w1 = np.ravel(weights[0:8, 0:8])
        part_w2 = np.ravel(weights[0:8, 8:16])
        part_w3 = np.ravel(weights[8:16, 0:8])
        part_w4 = np.ravel(weights[8:16, 8:16])
        weights_pieces = [part_w1, part_w2, part_w3, part_w4]

        '''
          Compute the histogram
          part1 = [10, 30, 12, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          part_w1 =[ 1.0, 300, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          data_p1 = [ 333, 0, 0, 0, 0, 0, 0, 0] // 1.0 + 300 + 32 + 0 
          
        '''

        data_p1 = np.zeros((8,))
        data_p2 = np.zeros((8,))
        data_p3 = np.zeros((8,))
        data_p4 = np.zeros((8,))
        histograms_pieces = [data_p1, data_p2, data_p3, data_p4]

        for _part in range(4):
            for _index in range(16):
                o = int(analysis.get_bin_pos(pieces[_part][_index]))
                w = weights_pieces[_part][_index]
                _k = analysis.get_8_bin_histogram_pos(o)
                histograms_pieces[_part][_k] += w

        k1 = histograms_pieces[0]
        k2 = histograms_pieces[1]
        k3 = histograms_pieces[2]
        k4 = histograms_pieces[3]
        key_desc = [x, y, k1 + k2 + k3 + k4]
        return key_desc

    def key_point_descriptors(self):
        final_list = []
        for octave in range(len(self.extrema)):
            for i in range(3):
                _sigma_ = self.get_sigma_for_hog(octave)
                final_list.append(self.compute_key_point_orientations(
                    self.extrema[octave][i, :, :],
                    self.grad[i],
                    self.mag[i], _sigma_[i]))
        return final_list

    @staticmethod
    def get_sigma_for_hog(octave_id):
        if octave_id == 0:
            return sigmas[0:3]
        if octave_id == 1:
            return sigmas[3:6]
        if octave_id == 2:
            return sigmas[6:9]
        if octave_id == 3:
            return sigmas[9:12]

    def compute_key_point_orientations(self, extrema, grad, mag, __sigma):
        descriptors = []
        window = self.get_gauss_window(WINDOW_SIZE, __sigma)
        for r in range(extrema.shape[0]):
            for c in range(extrema.shape[1]):
                if extrema[r][c] == 1:
                    try:
                        mat_extrema = self.sub_region(extrema, r, c)
                        mat_mag = self.sub_region(mag, r, c)
                        mat_ori = self.sub_region(grad, r, c)
                        if len(mat_extrema) == 0 or len(mat_mag) == 0 or len(mat_ori) == 0:
                            continue
                        weights = window * mat_extrema * mat_mag
                    except ValueError:
                        # print("KNOWN ISSUE :: Value error occurred due to edge problem which will be handled later")
                        continue
                    descriptors.append(self.compute_orientation_histogram(weights, mat_ori, r, c))
        return descriptors

    def compute(self):

        self.init_octaves()
        self.init_dogs()
        self.init_extrema()

        if not analysis.gen:
            print("Analysis is OFF, to turn on change gen=True in analysis2.py to see intermediate "
                  "images in output folder.")

        self.convolve_gauss()
        analysis.save(self.octave, '__octave__', 6)

        self.compute_dogs()
        analysis.print_data(self.dog, '__dog__', 5)

        self.init_extrema()
        self.detect_interest_points()
        analysis.print_extrema(self.extrema, '__extrma__', 3)

        self.apply_taylor()
        self.stats()

        self.comput_grad_mag()

        data = self.key_point_descriptors()
        repr(data)
        f = open('descriptors.json', 'w')
        f.write('data = ' + repr(data) + '\n')
        f.close()
        print("check the JSON file for descriptor  details descriptors.json")

        print("Output are plotted now")
        self.plot()

        # comparision against ORB
        img, kps = orb.opencv_sift(self.f)
        orb.show(img)
        print("Total detected points from ORB", kps)
        print("Total detected points from SIFT", self.total_count)

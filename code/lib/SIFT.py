# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '3'
import numpy as np
import cv2 as cv
import time
import copy
import math
from functools import cmp_to_key


def isExtremum(cube):
    if cube[1,1,1] > 0 and np.all(cube[1,1,1] >= cube[0:3, 0:3, :]):
        return True
    if cube[1,1,1] < 0 and np.all(cube[1,1,1] <= cube[0:3, 0:3, :]):
        return True
    return False

def computeGradient(cube):
    dy = (cube[2, 1, 1] - cube[0, 1, 1])/2
    dx = (cube[1, 2, 1] - cube[1, 0, 1])/2
    ds = (cube[1, 1, 2] - cube[1, 1, 0])/2
    return np.array([dy, dx, ds])

def computeHessian(cube):
    dxx = cube[1, 2, 1] - 2 * cube[1, 1, 1] + cube[1, 0, 1]
    dyy = cube[2, 1, 1] - 2 * cube[1, 1, 1] + cube[0, 1, 1]
    dss = cube[1, 1, 2] - 2 * cube[1, 1, 1] + cube[1, 1, 0]
    dxy = (cube[2, 2, 1] - cube[0, 2, 1] - cube[2, 0, 1] + cube[0, 0, 1]) / 4
    dxs = (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0]) / 4
    dys = (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0]) / 4
    return np.array([[dyy, dxy, dys], 
                    [dxy, dxx, dxs],
                    [dys, dxs, dss]])

def compare(kp1, kp2):
    if kp1.x != kp2.x:
        return kp1.x - kp2.x
    elif kp1.y != kp2.y:
        return kp1.y - kp2.y
    elif kp1.size != kp2.size:
        return kp2.size - kp1.size
    elif kp1.response != kp2.response:
        return kp2.response - kp1.response
    elif kp1.octave != kp2.octave:
        return kp2.octave - kp1.octave
    return 0

def equal(kp1, kp2):
    if kp1.x == kp2.x and kp1.y == kp2.y and kp1.size == kp2.size and \
        kp1.response == kp2.response and kp1.octave == kp2.octave:
        return True
    return False

def unpackOctave(keypoint):
    octave = keypoint.octave & 255
    s = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float(1 << octave) if octave >= 0 else float(1 << -octave)
    return octave, s, scale

class Keypoint:
    def __init__(self , x , y , orientation , octave , size , response):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.octave = octave
        self.size = size
        self.response = response

class SIFT:
    def __init__(self , sigma = 1.6 , num_intervals = 3,  \
                 contrast_threshold = 0.04 , eigenvalue_ratio = 10, border=5):
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = (eigenvalue_ratio + 1)**2 / eigenvalue_ratio
        self.threshold = int(127 * contrast_threshold / num_intervals )
        self.border=border

    def fit(self , img):
        print("--start fiting--")
        start_time = time.time()

        base_img = self.generate_base_image(img)
        gaussian_kernels = self.generateGaussianKernels()
        OctaveGaussians, OctaveDoGs = self.generateGaussianAndDoG(base_img, gaussian_kernels)
        keypoints = self.keypoint_localization(OctaveGaussians, OctaveDoGs)

        keypoints = self.remove_redundant_keypoint(keypoints)
        keypoints = self.keypoints_convertion(keypoints)

        descriptors = self.getDescriptors(keypoints, OctaveGaussians)

        total_time = time.time() - start_time
        print(f"Find {len(keypoints)} keypoints in total !")
        print(f"cost {total_time:0.2f} (s)")
        return keypoints, descriptors

    def generate_base_image(self, img):
        img = cv.cvtColor(img , cv.COLOR_BGR2GRAY).astype(np.float32)
        img = cv.resize(img , (0 , 0) , fx = 2 , fy = 2 , interpolation = cv.INTER_LINEAR)
        diff = np.sqrt(max(self.sigma**2 - 1, 0.01))
        return cv.GaussianBlur(img , (0 , 0) , sigmaX = diff , sigmaY = diff)
        
    def generateGaussianKernels(self):
        num_octave = self.num_intervals + 3
        gaussian_kernels = [self.sigma]
        k = 2**(1 / self.num_intervals)
        sigma_previous = gaussian_kernels[0]
        
        for i in range(1, num_octave):
            sigma_next = k * sigma_previous
            gaussian_kernels.append(math.sqrt(sigma_next ** 2 - sigma_previous ** 2))
            sigma_previous = sigma_next
        return gaussian_kernels

    def generateGaussianAndDoG(self, base_img, gaussian_kernels):
        num_octave = int((np.log2(min(base_img.shape))))-1
        previous_img = base_img
        OctaveGaussians, OctaveDoGs = [], []
        for i in range(num_octave):
            Gaussian_imgs , DoG_imgs = [previous_img] , []
            for gaussian_kernel in gaussian_kernels[1:]:
                img = cv.GaussianBlur(previous_img , (0 , 0) , sigmaX = gaussian_kernel , sigmaY = gaussian_kernel)
                Gaussian_imgs.append(img)
                DoG_imgs.append(img - previous_img)
                previous_img = img
            previous_img = Gaussian_imgs[self.num_intervals]
            previous_img = cv.resize(previous_img , (0 , 0), fx=1/2, fy=1/2, interpolation = cv.INTER_LINEAR)
            OctaveGaussians.append(Gaussian_imgs)
            OctaveDoGs.append(DoG_imgs)
        return OctaveGaussians, OctaveDoGs

    def keypoint_localization(self, OctaveGaussians, OctaveDoGs):
        keypoints = []
        count = 0
        for octave, DoGs in enumerate(OctaveDoGs):
            for s in range(1, self.num_intervals+1):
                h, w = DoGs[s].shape
                cat_img = np.stack((DoGs[s-1], DoGs[s], DoGs[s+1]), axis=2)

                for y in range(self.border, h-self.border):
                    for x in range(self.border, w-self.border):
                        if abs(cat_img[y,x,1]) > self.threshold:
                            cube = cat_img[y-1:y+2, x-1:x+2, :]
                            if isExtremum(cube):
                                result = self.accurate_keypoint_localization(DoGs, y, x, s, h, w, octave)
                                if result is not None:
                                    count += 1
                                    keypoint, local_s = result
                                    keypoints_with_orientations = self.computeKeypointsWithOrientations(keypoint, octave, OctaveGaussians[octave][local_s])
                                    for keypoint_with_orientation in keypoints_with_orientations:
                                        keypoints.append(keypoint_with_orientation)
                                    keypoints.append(keypoint)

                                    print(f"Find {len(keypoints)} keypoints({count})...", end="\r")
        
        return keypoints

    def accurate_keypoint_localization(self, DoGs, y, x, s, h, w, octave):
        
        converge = False
        cube, gradient, hessian = None, None, None
        for _ in range(5):
            cube = np.stack([DoGs[s-1][y-1:y+2, x-1:x+2], 
                                DoGs[s][y-1:y+2, x-1:x+2], 
                                DoGs[s+1][y-1:y+2, x-1:x+2]], axis=2) / 255
            # print(cube)
            gradient = computeGradient(cube)
            hessian = computeHessian(cube)
            offset = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if np.all(np.abs(offset) < 0.5):
                converge = True
                break
            y += int(np.round(offset[0]))
            x += int(np.round(offset[1]))
            s += int(np.round(offset[2]))
            if  (s < 1 or s >= self.num_intervals) or \
                (x < self.border or x >= w - self.border) or \
                (y < self.border or y >= h - self.border):
                return None
        if not converge:
            return None
        
        contrast_response = cube[1,1,1] + 0.5 * np.dot(gradient , offset)
        if abs(contrast_response) * self.num_intervals >= self.contrast_threshold:
            trace = hessian[0, 0] + hessian[1, 1]
            det = (hessian[0, 0] * hessian[1, 1]) - (hessian[0, 1] * hessian[1, 0]) + 1e-8
            edge_response = trace**2 / det
            if det > 1e-8 and edge_response < self.edge_threshold:
                keypoint = Keypoint(
                    x = (x + offset[1]) * 2**octave , 
                    y = (y + offset[0]) * 2**octave , 
                    orientation = None , 
                    octave = octave + 256 * s + 65536 * int(np.round(255 * (offset[0] + 0.5))) , 
                    size = self.sigma * 2**((s + offset[2]) / self.num_intervals) * (2**(octave + 1)) , 
                    response = abs(contrast_response)
                )
                return keypoint , s
        return None
    
    def computeKeypointsWithOrientations(self, keypoint, octave, gaussian_img, 
                        radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        keypoints_with_orientations = []
        h, w = gaussian_img.shape

        scale = scale_factor * keypoint.size / (2 ** (octave + 1))
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        x = int(np.round(keypoint.x / 2**octave))
        y = int(np.round(keypoint.y / 2**octave))
        for i in range(-radius , radius + 1):
            for j in range(-radius , radius + 1):
                if x + i <= 0 or x + i >= w - 1 or y + j <= 0 or y + j >= h - 1:
                    continue
                dx = gaussian_img[y + j , x + i + 1] - gaussian_img[y + j , x + i - 1]
                dy = gaussian_img[y + j + 1 , x + i] - gaussian_img[y + j - 1 , x + i]
                gradient_magnitude = np.sqrt(dx**2 + dy**2)
                gradient_degree = np.rad2deg(np.arctan2(dy , dx))
                weight = np.exp(weight_factor * (i**2 + j**2))
                raw_histogram[int(np.round(gradient_degree / (360 / num_bins)))] += weight * gradient_magnitude
        
        smooth_weight = [1, 4, 6, 4, 1]
        for i in range(num_bins):
            for j in range(5):
                smooth_histogram[i] += raw_histogram[(i - 2 + j)%num_bins] * smooth_weight[j] 
            smooth_histogram[i] /= 16
        
        peak_idxs = []
        for i in range(num_bins):
            if smooth_histogram[i] > smooth_histogram[i-1] and smooth_histogram[i] > smooth_histogram[(i+1)%num_bins]:
                peak_idxs.append(i)

        orientation_max = max(smooth_histogram)

        for idx in peak_idxs:
            peak_value = smooth_histogram[idx]
            if peak_value >= peak_ratio * orientation_max:
                left_value = smooth_histogram[(idx - 1) % num_bins]
                right_value = smooth_histogram[(idx + 1) % num_bins]
                interpolated_peak_index = (idx + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < 1e-7:
                    orientation = 0
                new_keypoint = copy.deepcopy(keypoint)
                new_keypoint.orientation = orientation
                keypoints_with_orientations.append(new_keypoint)
        return keypoints_with_orientations

    def remove_redundant_keypoint(self, keypoints):
        keypoints.sort(key=cmp_to_key(compare))
        cleaned_keypoints = [keypoints[0]]
        for keypoint in keypoints[1:]:
            if not equal(keypoint, cleaned_keypoints[-1]):
                cleaned_keypoints.append(keypoint)
        return cleaned_keypoints

    def keypoints_convertion(self, keypoints):
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.x /= 2
            keypoint.y /= 2
            keypoint.size /= 2
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def getDescriptors(self, keypoints, OctaveGaussians, \
            window_size=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):

        descriptors = []
        weight_multiplier = -0.5 / ((0.5 * window_size) ** 2)
        bins_per_degree = num_bins / 360.

        for keypoint in keypoints:
            octave, layer, scale = unpackOctave(keypoint)
            Gaussian_img = OctaveGaussians[octave + 1][layer]
            num_rows, num_cols = Gaussian_img.shape
            x = int(np.round(scale * keypoint.x))
            y = int(np.round(scale * keypoint.y))
            degree = 360 - keypoint.orientation
            cos_deg = np.cos(np.deg2rad(degree))
            sin_deg = np.sin(np.deg2rad(degree))

            bin_info_list = []
            histogram = np.zeros((window_size + 2, window_size + 2, num_bins))

            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            radius = int(round(hist_width * np.sqrt(2) * (window_size + 1) * 0.5))
            radius = int(min(radius, np.sqrt(num_rows ** 2 + num_cols ** 2)))

            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    rotat_j = j * sin_deg + i * cos_deg
                    rotat_i = j * cos_deg - i * sin_deg

                    bin_j = (rotat_j / hist_width) + 0.5 * window_size - 0.5
                    bin_i = (rotat_i / hist_width) + 0.5 * window_size - 0.5

                    if bin_j > -1 and bin_j < window_size and bin_i > -1 and bin_i < window_size:
                        window_j = int(round(y + j))
                        window_i = int(round(x + i))
                        if window_j > 0 and window_j < num_rows - 1 and window_i > 0 and window_i < num_cols - 1:
                            dx = Gaussian_img[window_j, window_i + 1] - Gaussian_img[window_j, window_i - 1]
                            dy = Gaussian_img[window_j + 1, window_i] - Gaussian_img[window_j - 1, window_i]
                            gradient_magnitude = np.sqrt(dx**2 + dy**2)
                            gradient_degree = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((rotat_i / hist_width) ** 2 + (rotat_j / hist_width) ** 2))
                            
                            bin_info_list.append((
                                bin_j,
                                bin_i,
                                weight * gradient_magnitude,
                                (gradient_degree - degree) * bins_per_degree
                            ))
            
            for y_bin, x_bin, magnitude, deg_bin in bin_info_list:
                #  trilinear interpolation

                y_bin_f, y_bin_c = int(np.floor(y_bin)), int(np.floor(y_bin))+1
                x_bin_f, x_bin_c = int(np.floor(x_bin)), int(np.floor(x_bin))+1
                deg_bin_f, deg_bin_c = int(np.floor(deg_bin)) % num_bins, (int(np.floor(deg_bin))+1) % num_bins

                y_ratio = y_bin - y_bin_f
                x_ratio = x_bin - x_bin_f
                deg_ratio = deg_bin - int(np.floor(deg_bin))

                c1 = magnitude * y_ratio
                c0 = magnitude * (1 - y_ratio)
                c11 = c1 * x_ratio
                c10 = c1 * (1 - x_ratio)
                c01 = c0 * x_ratio
                c00 = c0 * (1 - x_ratio)
                c111 = c11 * deg_ratio
                c110 = c11 * (1 - deg_ratio)
                c101 = c10 * deg_ratio
                c100 = c10 * (1 - deg_ratio)
                c011 = c01 * deg_ratio
                c010 = c01 * (1 - deg_ratio)
                c001 = c00 * deg_ratio
                c000 = c00 * (1 - deg_ratio)

                histogram[y_bin_f + 1, x_bin_f + 1, deg_bin_f] += c000
                histogram[y_bin_f + 1, x_bin_f + 1, deg_bin_c] += c001
                histogram[y_bin_f + 1, x_bin_c + 1, deg_bin_f] += c010
                histogram[y_bin_f + 1, x_bin_c + 1, deg_bin_c] += c011
                histogram[y_bin_c + 1, x_bin_f + 1, deg_bin_f] += c100
                histogram[y_bin_c + 1, x_bin_f + 1, deg_bin_c] += c101
                histogram[y_bin_c + 1, x_bin_c + 1, deg_bin_f] += c110
                histogram[y_bin_c + 1, x_bin_c + 1, deg_bin_c] += c111

            descriptor = histogram[1 : -1 , 1 : -1 , : ].reshape(-1)
            threshold = descriptor_max_value * np.linalg.norm(descriptor)
            descriptor[descriptor > threshold] = threshold
            descriptor /= max(np.linalg.norm(descriptor) , 1e-7)
            descriptor = np.round(512 * descriptor)
            descriptor[descriptor < 0] = 0
            descriptor[descriptor > 255] = 255
            descriptors.append(descriptor)
        descriptors = np.array(descriptors)
        return descriptors

# using example
img = cv.imread("/tmp2/b07902058/DVE_hw2/image_stitching/code/lib/8.jpg")
print(img.shape)
sift = SIFT()
keypoints, descriptors = sift.fit(img)

# draw keypoint
for kp in keypoints:
    image = cv.circle(img, (int(kp.x), int(kp.y)), radius=2, color=(0, 0, 255), thickness=2)

cv.imwrite("test.jpg", image)
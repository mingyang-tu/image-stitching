# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '3'
import numpy as np
import cv2 as cv
import time
import copy
import math
from multiprocessing import Pool


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
        self.threshold = int(255 * contrast_threshold / num_intervals / 2)
        self.border=border

    def fit(self , img):
        print("--start fiting--")
        start_time = time.time()

        base_img = self.generate_base_image(img)
        gaussian_kernels = self.generateGaussianKernels()
        OctaveGaussians, OctaveDoGs = self.generateGaussianAndDoG(base_img, gaussian_kernels)
        
        # for i, Gaussians in enumerate(OctaveGaussians):
        #     for j, Gaussian in enumerate(Gaussians):
        #         cv.imwrite(f"./Octave_{i}/Gaussian/Gaussian_{j}.jpg", Gaussian)
        # for i, OctaveDoG in enumerate(OctaveDoGs):
        #     for j, Dog in enumerate(OctaveDoG):
        #         cv.imwrite(f"./Octave_{i}/DoG/DoG_{j}.jpg", Dog)
        keypoints = self.keypoint_localization(OctaveGaussians, OctaveDoGs)

        total_time = time.time() - start_time
        print(f"Find {len(keypoints)} keypoints in total !")
        print(f"cost {total_time:0.2f} (s)")
        return keypoints
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
                                    # keypoints_with_orientations = self.computeKeypointsWithOrientations(keypoint, octave, OctaveGaussians[octave][local_s])
                                    # for keypoint_with_orientation in keypoints_with_orientations:
                                    #     keypoints.append(keypoint_with_orientation)
                                    keypoints.append(keypoint)

                                    print(f"Find {len(keypoints)} keypoints({count})...", end="\r")
        return keypoints

    def accurate_keypoint_localization(self, DoGs, y, x, s, h, w, octave):
        
        converge = False
        cube, gradient, hessian = None, None, None
        for _ in range(5):
            cat_img = np.stack([DoGs[s-1], DoGs[s], DoGs[s+1]], axis=2)
            cube = cat_img[y-1:y+2, x-1:x+2, :] / 255
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
                    size = self.sigma * 2**((s + offset[2]) / self.num_intervals) * 2**(octave + 1) , 
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


# using example
img = cv.imread("/tmp2/b07902058/DVE_hw2/memorial0064.png")

sift = SIFT()
keypoints = sift.fit(img)

# draw keypoint
for kp in keypoints:
    image = cv.circle(img, (int(kp.x/2), int(kp.y/2)), radius=2, color=(0, 0, 255), thickness=2)

cv.imwrite("test.jpg", image)
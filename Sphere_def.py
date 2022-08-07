import cv2
import os
import time
import glob
import pandas as pd
import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu


class SphereImages:
    def __init__(self, window_, m, n):
        self.window = window_
        self.m = m
        self.n = n
        self.area, self.condition = 0, 0
        self.path_f, self.markers = None, None
        self.symbol, self.blurred = None, None
        self.contours, self.root_result = [], None

    def bytes_(self, img_):
        ima = cv2.resize(img_, (self.m, self.n))
        return cv2.imencode('.png', ima)[1].tobytes()

    def update_dir(self, path):
        path_s = path.split('/')
        cad, self.path_f = len(path_s), path_s[0]
        for p in range(1, cad):
            self.path_f += '\\' + path_s[p]
        return self.path_f

    def load_image_i(self, orig, i, type_, filenames, id_sys):
        self.symbol = '\\' if id_sys == 0 else '/'
        if len(filenames) == 0:
            filenames = [img for img in glob.glob(orig + '*' + type_)]
            filenames.sort()
        if i < len(filenames):
            name = filenames[i]
            parts = name.split(self.symbol)
            name_i = parts[len(parts) - 1]
            image_ = cv2.imread(name)
        else:
            image_, name_i = [], []
        return filenames, image_, name_i, len(filenames)

    def preprocessing(self, img_):
        image_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        clh = cv2.createCLAHE(clipLimit=5)
        clh_img = clh.apply(image_gray_)
        self.blurred = cv2.GaussianBlur(clh_img, (5, 5), 0)

    def binary_ima(self, img, h_param):
        # Normalize image
        norm_img = cv2.normalize(img, None, alpha=-0.1, beta=1.1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_ = (255 * norm_img).astype(np.uint8)
        self.preprocessing(img_)
        # Adaptive threshold
        thresh = threshold_otsu(self.blurred)
        l_param = 8 if thresh < 150 else 5
        binary = cv2.adaptiveThreshold(self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, h_param, l_param)
        # Morphological operations
        arr = binary > 0
        binary = morphology.remove_small_objects(arr, min_size=10, connectivity=1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=2)
        arr = binary > 0
        markers = morphology.remove_small_objects(arr, min_size=2500, connectivity=1).astype(np.uint8)
        markers = morphology.remove_small_holes(markers.astype(np.bool_), area_threshold=7000, connectivity=1)
        self.markers = markers.astype(np.uint8)

    def verify_contour(self, contour):
        _, _, w, h = cv2.boundingRect(contour)
        self.condition = np.round(min(w, h) / max(w, h), 2)
        if self.condition > 0.6:
            return True
        else:
            return False

    def compute_area_rect(self, points, conv_value):
        left, right = min(points, key=lambda p: p[0]), max(points, key=lambda p: p[0])
        bottom, top = min(points, key=lambda p: p[1]), max(points, key=lambda p: p[1])
        # area
        base = (right[0] - left[0]) / conv_value
        height = (top[1] - bottom[1]) / conv_value
        self.area = np.round(base * height, 2)

    def spheres_ima(self, img_, conv_value):
        img_out = np.copy(img_)
        self.contours = cv2.findContours(self.markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = self.contours[0] if len(self.contours) == 2 else self.contours[1]
        self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)
        area_ref = np.round(img_.shape[0] * img_.shape[1] * 0.25, 2)
        radius_a, area_a = [], []
        for c in self.contours:
            area = cv2.contourArea(c)
            if area > 250:
                if self.verify_contour(c):
                    (x_, y_), radius_ = cv2.minEnclosingCircle(c)
                    (x_, y_), radius_ = (int(np.round(x_)), int(np.round(y_))), int(np.round(radius_))
                    area_c = np.round(np.pi * radius_ ** 2)
                    if area_c < area_ref:
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        self.compute_area_rect(box, conv_value)
                        cv2.drawContours(img_out, [box], 0, (0, 0, 255), 2)
                        cv2.circle(img_out, (x_, y_), radius_, (35, 255, 12), 3)
                        radius_a.append(np.round(radius_/conv_value, 3))
                        area_a.append(self.area)
        return img_out, radius_a, area_a

    def sphere_main(self, img_, ide_ima, h_param, results, conv_value):
        tic = time.time()
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        self.binary_ima(img_, h_param)
        ima_out, radius_, area_ = self.spheres_ima(img, conv_value)
        toc = np.round(time.time() - tic, 2)
        print(f'Time processing    : {toc} sec.')
        print(f'Number of spheres  : {len(radius_)}')
        area_total = np.round((img_.shape[0] * img_.shape[1]) / conv_value, 2)
        area_detected = 0
        n_spheres = len(radius_)
        for i in range(n_spheres):
            area_detected += area_[i]
            print(f' *** Cell No. {i + 1} ----> Radio: {radius_[i]} ----> Internal Area: {area_[i]}')
            # save results
            new_row = pd.DataFrame.from_records([{'Image': ide_ima, 'Sphere': i+1, 'Radius (um)': radius_[i],
                                                 'Area (um2)': area_[i], 'Time (sec)': toc}])
            results = pd.concat([results, new_row], ignore_index=True)

        percentage = np.round((area_detected / area_total) * 100, 2)

        return ima_out, results, area_total, area_detected, percentage, n_spheres, toc

    def save_csv_file(self, results, path_des, name_file):
        # Save data in csv file
        self.root_result = os.path.join(path_des, name_file + '.csv')
        results.to_csv(self.root_result, index=False)
        print('----------------------------------------------')
        print('..... Save data in CSV file successfully .....')
        print('----------------------------------------------')







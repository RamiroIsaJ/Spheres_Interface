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
        self.m, self.n = m, n
        self.area, self.condition = 0, 0
        self.path_f, self.markers = None, None
        self.symbol, self.blurred, self.filters, self.big_contour = None, None, [], []
        self.contours, self.root_file, self.root_ima = [], None, None

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
        clh = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10, 10))
        clh_img = clh.apply(image_gray_)
        self.blurred = cv2.GaussianBlur(clh_img, (5, 5), 0)

    def build_filters(self):
        filters_, k_size, sigma = [], 21, 2.0
        for theta in np.arange(0, np.pi, np.pi / 4):
            kern = cv2.getGaborKernel((k_size, k_size), sigma, theta, 10.0, 0.50, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            self.filters.append(kern)

    def apply_gabor(self):
        gabor_img_ = np.zeros_like(self.blurred)
        for kern in self.filters:
            np.maximum(gabor_img_, cv2.filter2D(self.blurred, cv2.CV_8UC3, kern), gabor_img_)
        return gabor_img_

    def dist(self, xp, yp):
        return np.sqrt(np.sum((xp - yp) ** 2))

    def calculate_contour(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.big_contour, areas_c = [], []
        for c in contours:
            if cv2.contourArea(c) > 300:
                self.big_contour.append(c)
                areas_c.append(cv2.contourArea(c))
        return np.array(areas_c)

    def p_circle(self, binary_):
        _ = self.calculate_contour(binary_)
        contour = sorted(self.big_contour, key=cv2.contourArea, reverse=True)
        (x_, y_), radius_ = cv2.minEnclosingCircle(contour[0])
        return int(x_), int(y_), int(radius_)

    def well_ima(self, img_, gabor_img_, conv_value_):
        area_, total_well = 0, 0
        thresh1 = cv2.threshold(gabor_img_, threshold_otsu(gabor_img_), 255, cv2.THRESH_TOZERO_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        marker = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=2)
        arr1 = marker > 0
        marker = morphology.remove_small_objects(arr1, min_size=5000, connectivity=1).astype(np.uint8)
        marker = morphology.remove_small_holes(marker.astype(np.bool_), area_threshold=8000, connectivity=1)
        marker = marker.astype(np.uint8)
        # compute well area
        x_, y_, radius_ = self.p_circle(marker)
        roi_ima = np.zeros_like(marker, dtype=np.uint8)
        cv2.circle(roi_ima, (x_, y_), radius_, 1, -1)
        vals_ = np.where(roi_ima == 1)
        marker1 = np.zeros_like(roi_ima, dtype=np.uint8)
        marker1[vals_] = marker[vals_]
        # complement analysis
        marker2 = cv2.bitwise_not(marker.astype(np.uint8))
        # full region
        markers = cv2.bitwise_or(marker1, marker2)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        markers = cv2.morphologyEx(markers, cv2.MORPH_ERODE, kernel, iterations=3)
        markers = markers.astype(np.uint8)
        section1, section2 = markers[y_, 0:x_], markers[y_, x_:]
        idx1, idx2 = np.where(section1 == 254)[0], np.where(section2 == 254)[0]
        if len(idx1) > 1 and len(idx2) > 1:
            idx1, idx2 = np.where(section1 == 254)[0][-1], np.where(section2 == 254)[0][0] + x_
            x_1 = int((idx2 + idx1) / 2)
            radius_1 = int((idx2 - idx1) / 2)
            binary = cv2.morphologyEx(marker, cv2.MORPH_ERODE, kernel, iterations=3)
            arr1 = binary > 0
            binary = morphology.remove_small_objects(arr1, min_size=8000, connectivity=1).astype(np.uint8)
            binary = morphology.remove_small_holes(binary.astype(np.bool_), area_threshold=45000, connectivity=1)
            binary = binary.astype(np.uint8)
            roi_ima = np.zeros_like(binary, dtype=np.uint8)
            cv2.circle(roi_ima, (x_1, y_), radius_1, 1, -1)
            vals_ = np.where(roi_ima == 1)
            total_detected = np.sum(binary[vals_] == 1)
            total_well = np.sum(roi_ima[vals_] == 1)
            percentage_ = np.round((total_detected / total_well) * 100, 3)
            roi_ima = np.zeros_like(binary, dtype=np.uint8)
            roi_ima[vals_] = binary[vals_]
            _ = self.calculate_contour(roi_ima)
            image_r_ = np.copy(img_)
            for c in self.big_contour:
                area_ = cv2.contourArea(c)
                if area_ > 250:
                    cv2.drawContours(image_r_, c, -1, (0, 0, 255), 3)
            cv2.circle(image_r_, (x_1, y_), radius_1, (255, 0, 0), 3)
            rel1, rel2 = radius_1 / image_r_.shape[0], radius_1 / image_r_.shape[1]
            if rel1 < 0.40 and rel2 < 0.40:
                percentage_, image_r_ = -1, 0
        else:
            percentage_, image_r_ = -1, 0
        return image_r_, percentage_, area_/conv_value_, total_well/conv_value_

    def verify_regions(self, binary_):
        total_ = binary_.shape[0] * binary_.shape[1]
        num_ones = np.sum(binary_.flatten() == 1)
        return np.round((num_ones / total_), 2)

    def binary_ima1(self, gabor_img_):
        thresh1 = threshold_otsu(gabor_img_)
        val_max = 70 if thresh1 <= 77 else 80
        val_iter = 2 if thresh1 <= 77 else 3
        binary = np.array(gabor_img_ <= val_max).astype(np.uint8)
        relation_ = self.verify_regions(binary)
        if relation_ >= 0.62:
            binary = np.array(gabor_img_ <= 90).astype(np.uint8)
            binary = 1 - binary
        areas_contours = self.calculate_contour(binary)
        min_area, max_area = int(2.0*np.min(areas_contours)), np.max(areas_contours)
        min_area = 800 if max_area < 700 else 1000
        max_area = 7000
        # Morphological operations
        arr = binary > 0
        binary = morphology.remove_small_objects(arr, min_size=10, connectivity=1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=int(val_iter))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=2)
        arr = binary > 0
        markers = morphology.remove_small_objects(arr, min_size=min_area, connectivity=1).astype(np.uint8)
        markers = morphology.remove_small_holes(markers.astype(np.bool_), area_threshold=int(max_area), connectivity=1)
        self.markers = markers.astype(np.uint8)

    def binary_ima2(self, thresh_, h_param_):
        param = 8 if thresh_ < 145 else 5
        binary = cv2.adaptiveThreshold(self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, h_param_, param)
        # Morphological operations
        arr = binary > 0
        binary = morphology.remove_small_objects(arr, min_size=10, connectivity=1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=3)
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

    def spheres_ima(self, img_, conv_value_):
        img_out = np.copy(img_)
        self.contours = cv2.findContours(self.markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = self.contours[0] if len(self.contours) == 2 else self.contours[1]
        self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)
        area_ref = np.round(img_.shape[0] * img_.shape[1] * 0.25, 2)
        area_total_ = np.round(((img_.shape[0] / conv_value_) * (img_.shape[1] / conv_value_)), 2)
        area_a, ide = [], 1
        for c in self.contours:
            area = cv2.contourArea(c)
            if area > 250:
                if self.verify_contour(c):
                    (x_, y_), radius_ = cv2.minEnclosingCircle(c)
                    (x_, y_), radius_ = (int(np.round(x_)), int(np.round(y_))), int(np.round(radius_))
                    area_c = np.round(np.pi * radius_ ** 2)
                    if area_c < area_ref:
                        cv2.drawContours(img_out, c, -1, (0, 0, 255), 3)
                        cv2.circle(img_out, (x_, y_), radius_, (35, 255, 12), 3)
                        cx, cy = int(x_ - 5), int(y_ - radius_)
                        cv2.putText(img_out, str(ide), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        area_a.append(area / conv_value_)
                        ide += 1
        return img_out, area_total_, area_a

    def sphere_main(self, img_, ide_ima, h_param, results, conv_value):
        tic = time.time()
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        self.preprocessing(img_)
        gabor_img = self.apply_gabor()
        ima_out, percentage, area_, area_total = self.well_ima(img_, gabor_img, conv_value)
        thresh = threshold_otsu(self.blurred)
        print(percentage, thresh, threshold_otsu(gabor_img))
        if percentage < 0:
            if threshold_otsu(gabor_img) >= 73 and thresh <= 128:
                self.binary_ima1(gabor_img)
            else:
                self.binary_ima2(thresh, h_param)
            ima_out, area_total, area_ = self.spheres_ima(img, conv_value)
            toc = np.round(time.time() - tic, 2)
            print(f'Time processing    : {toc} sec.')
            print(f'Number of spheres  : {len(area_)}')
            area_detected, n_spheres = 0, len(area_)
            for i in range(n_spheres):
                area_detected += area_[i]
                per = np.round(((area_[i] * 100) / area_total), 2)
                print(f' *** Cell No. {i + 1} ----> Internal Area: {area_[i]}')
                # save results
                new_row = pd.DataFrame.from_records([{'Image': ide_ima, 'Sphere': i+1, 'Detected Area (um2)': area_[i],
                                                      'Percentage Area': per, 'Image Area (um2)': area_total,
                                                      'Time (sec)': toc}])
                results = pd.concat([results, new_row], ignore_index=True)
            area_detected = np.round(area_detected, 2)
            percentage = np.round(((area_detected * 100) / area_total), 2)
        else:
            n_spheres, area_detected = 1, np.copy(area_)
            toc = np.round(time.time() - tic, 2)
            # save results
            new_row = pd.DataFrame.from_records([{'Image': ide_ima, 'Sphere': 1, 'Detected Area (um2)': area_,
                                                  'Percentage Area': percentage, 'Image Area (um2)': area_total,
                                                  'Time (sec)': toc}])
            results = pd.concat([results, new_row], ignore_index=True)
        return ima_out, results, area_total, area_detected, percentage, n_spheres, toc

    def save_image_out(self, ima_out_, path_des, name_ima):
        self.root_ima = os.path.join(path_des, name_ima)
        cv2.imwrite(self.root_ima, ima_out_)
        print('..... Image saved successfully .....')

    def save_csv_file(self, results, path_des, name_file):
        # Save data in csv file
        self.root_file = os.path.join(path_des, name_file + '.csv')
        results.to_csv(self.root_file, index=False)
        print('----------------------------------------------')
        print('..... Save data in CSV file successfully .....')
        print('----------------------------------------------')

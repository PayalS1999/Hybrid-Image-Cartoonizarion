# This is a sample Python script.
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import os
# Press Ctrl+R to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
if __name__ == '__main__':
    input_dir  = '/home/sriramg/payalsaha/StyleID/data/llm-cnt'
    output_dir = '/home/sriramg/payalsaha/image-filtering/output'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        fname = os.path.join(input_dir, filename)
        og_img = cv2.imread(fname, cv2.IMREAD_COLOR)
        #gray = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(og_img, (0, 0), 5.6)
        lap = cv2.Laplacian(blur, cv2.CV_64F, ksize=5)
        abs_lap = cv2.convertScaleAbs(lap)
        sharpen = cv2.addWeighted(blur, 1.0, og_img, 0.25, 0)
        #edges_bgr = cv2.cvtColor(abs_lap, cv2.COLOR_GRAY2BGR)
        #_, mask = cv2.threshold(edges_bgr, 20, 255, cv2.THRESH_BINARY)
        #color_mask = np.zeros_like(og_img)

        cv2.imwrite(os.path.join(output_dir, filename), sharpen)
        print("Saved {}".format(filename))

### Gaussian filtering


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

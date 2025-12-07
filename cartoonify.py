import cv2
import numpy as np
import os


def cartoonify(image_path, output_path=None):
    # Step 1: Read & Resize Image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))

    # Step 2: Smooth Image (bilateral filter preserves edges)
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 3: Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 7), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)

    # Step 4: Color Simplification (Posterization using K-means)
    data = smoothed.reshape((-1, 3))
    data = np.float32(data)

    K = 8  # Number of color clusters
    _, labels, centers = cv2.kmeans(data, K, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    10, cv2.KMEANS_RANDOM_CENTERS)

    quantized = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)

    # Step 5: Combine Edges and Simplified Colors
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quantized, edges_colored)

    # Step 6: Optional – enhance brightness for a more child-like look
    cartoon = cv2.convertScaleAbs(cartoon, alpha=1.1, beta=10)

    return cartoon

if __name__ == '__main__':
    input_dir  = '/home/sriramg/payalsaha/StyleID/data/llm-cnt'
    output_dir = '/home/sriramg/payalsaha/image-filtering/output/cartoonify'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        fname = os.path.join(input_dir, filename)
        cartoon = cartoonify(fname, output_path= output_dir)
        cv2.imwrite(os.path.join(output_dir, filename), cartoon)
        print('Saved {}'.format(filename))
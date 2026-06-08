import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR,  'data', 'images')

deleted = 0
for f in os.listdir(IMG_DIR):
    path = os.path.join(IMG_DIR, f)
    img  = cv2.imread(path)
    if img is None: continue
    brightness = np.mean(img)
    sharpness  = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if brightness < 80 or sharpness < 100:
        os.remove(path)
        deleted += 1

print(f"Deleted: {deleted}")
print(f"Remaining: {len(os.listdir(IMG_DIR))}")
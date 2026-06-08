import os, cv2, numpy as np
from matplotlib import pyplot as plt

IMG_DIR = r'C:\\Users\\chsur\\projects\\face\\vgg16-same-size\\up\\data\\images'
LBL_DIR = r'C:\\Users\\chsur\\projects\\face\\vgg16-same-size\\up\\data\\labels'

files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
labels = set(f.rsplit('.', 1)[0] for f in os.listdir(LBL_DIR) if f.endswith('.json'))

dark, blurry, good = [], [], []

for f in files:
    path = os.path.join(IMG_DIR, f)
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(img)
    sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()

    if brightness < 80:
        dark.append(f)
    elif sharpness < 100:
        blurry.append(f)
    else:
        good.append(f)

with_face    = [f for f in good if f.rsplit('.', 1)[0] in labels]
without_face = [f for f in good if f.rsplit('.', 1)[0] not in labels]

print(f"Total images  : {len(files)}")
print(f"Dark          : {len(dark)}")
print(f"Blurry        : {len(blurry)}")
print(f"Good          : {len(good)}")
print(f"  With face   : {len(with_face)}")
print(f"  No face     : {len(without_face)}")
print(f"No-face ratio : {len(without_face)/len(good)*100:.1f}%")
print()

# How many more you need
target = 150
needed = max(0, target - len(good))
print(f"Target        : {target} good images")
print(f"Need to add   : {needed} more images")

# Preview 12 good images
sample = good[:12]
fig, axes = plt.subplots(3, 4, figsize=(20,12))
for idx, fname in enumerate(sample):
    img = cv2.imread(os.path.join(IMG_DIR, fname))
    axes.flatten()[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes.flatten()[idx].set_title('face' if fname.rsplit('.', 1)[0] in labels else 'no-face')
    axes.flatten()[idx].axis('off')
plt.tight_layout()
plt.show()
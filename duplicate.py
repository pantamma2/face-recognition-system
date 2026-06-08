import os, cv2, numpy as np

IMG_DIR = r'C:\\Users\\chsur\\projects\\face\\vgg16-same-size\\up\\data\\images'
files  = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
thumbs = [(f, cv2.resize(cv2.imread(os.path.join(IMG_DIR,f)), (16,16)).flatten().astype(np.float32)) for f in files]

THRESHOLD = 0.99
seen, keep, dupes = [], [], []

for fname, vec in thumbs:
    if any(np.dot(vec,s)/(np.linalg.norm(vec)*np.linalg.norm(s)) > THRESHOLD for s in seen):
        dupes.append(fname)
    else:
        seen.append(vec)
        keep.append(fname)

print(f"Total: {len(files)} | Keep: {len(keep)} | Duplicates: {len(dupes)}")


# Delete duplicate images
for fname in dupes:
    os.remove(os.path.join(IMG_DIR, fname))
print(f"Deleted {len(dupes)} duplicates")

# Delete unwanted labels
LBL_DIR = r'C:\\Users\\chsur\\projects\\face\\vgg16-same-size\\up\\data\\labels'
images = set(f.rsplit('.', 1)[0] for f in os.listdir(IMG_DIR) if f.endswith('.jpg'))
labels = set(f.rsplit('.', 1)[0] for f in os.listdir(LBL_DIR) if f.endswith('.json'))

orphans = labels - images
for name in orphans:
    os.remove(os.path.join(LBL_DIR, name + '.json'))

print(f"Deleted {len(orphans)} unwanted labels")
print(f"Images : {len(os.listdir(IMG_DIR))}")
print(f"Labels : {len(os.listdir(LBL_DIR))}")
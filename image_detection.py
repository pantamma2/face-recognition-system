import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt
from PIL import Image

def build_model():
    inp = Input(shape=(120,120,3))
    vgg = VGG16(include_top=False)
    vgg.trainable = False
    f   = GlobalMaxPooling2D()(vgg(inp, training=False))
    c   = Dense(1, activation='sigmoid')(Dropout(0.3)(Dense(2048, activation='relu')(f)))
    r   = Dense(4, activation='sigmoid')(Dropout(0.3)(Dense(2048, activation='relu')(f)))
    return Model(inputs=inp, outputs=[c, r])

facetracker = build_model()
facetracker.load_weights(r'C:\\Users\\chsur\\projects\\face\\vgg16-same-size\\up\\best_facetracker.h5')
mtcnn = MTCNN(keep_all=False, device='cpu', min_face_size=40)


IMAGE =  'WhatsApp Image 2026-06-08 at 14.00.06.jpeg'

frame = cv2.imread(str(IMAGE))
if frame is None:
    raise FileNotFoundError(f'Could not open image: {IMAGE}')

h, w = frame.shape[:2]
display = frame.copy()

boxes, probs = mtcnn.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

if boxes is not None:
    x1,y1,x2,y2 = [int(v) for v in boxes[0]]
    pad  = int(0.2 * max(x2-x1, y2-y1))
    crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
    inp  = np.expand_dims(tf.image.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (120,120)) / 255.0, 0)
    conf = float(facetracker.predict(inp, verbose=0)[0][0][0])
    cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(display, f'face {conf:.2f}', (x1, max(y1-8,10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    title = f'FACE — {conf:.2f}'
else:
    title = 'NO FACE'

plt.figure(figsize=(7,5))
plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
plt.title(title); plt.axis('off'); plt.show()
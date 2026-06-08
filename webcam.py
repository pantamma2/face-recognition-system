
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16
from facenet_pytorch import MTCNN
from PIL import Image

# ── Build VGG16 model ────────────────────────────────────
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
print("VGG16 loaded!")

# ── Load MTCNN ───────────────────────────────────────────
mtcnn = MTCNN(keep_all=False, device='cpu', min_face_size=40)
print("MTCNN loaded!")
print("Press Q to quit")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(5): cap.read()   # warmup

while True:
    ret, frame = cap.read()
    if not ret: continue

    display = frame.copy()
    h, w    = frame.shape[:2]

    # ── MTCNN detection ──────────────────────────────────
    rgb_pil      = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(rgb_pil)

    if boxes is not None and len(boxes) > 0:
        best_idx     = np.argmax(probs)
        x1, y1, x2, y2 = boxes[best_idx]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop with padding for VGG16
        pad = int(0.2 * max(x2-x1, y2-y1))
        x1p = max(0, x1-pad)
        y1p = max(0, y1-pad)
        x2p = min(w, x2+pad)
        y2p = min(h, y2+pad)
        crop = frame[y1p:y2p, x1p:x2p]

        # ── VGG16 confidence ─────────────────────────────
        if crop.size > 0:
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp  = np.expand_dims(tf.image.resize(rgb, (120,120)) / 255.0, 0)
            conf, _ = facetracker.predict(inp, verbose=0)
            conf = float(conf[0][0])
        else:
            conf = float(probs[best_idx])

        # Draw single green box with VGG16 confidence
        cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(display, f'face {conf:.2f}',
                    (x1, max(y1-8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(display, 'no face', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow('MTCNN + VGG16  Q=quit', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
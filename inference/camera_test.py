import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image

# CONFIG

model_path = r"D:\Facial_Emotion\models\checkpoint_ResNet18\run_20260219_193106\best_epoch_2_valacc_70.86%.pth"
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMA_ALPHA = 0.7            # สูง = นิ่งขึ้น
CONF_THRESHOLD = 0.45      # ความมั่นใจขั้นต่ำ
SWITCH_FRAMES = 5          # ต้องชนะกี่เฟรมติดก่อนเปลี่ยน

print("Using device:", device)


# CLASS NAMES
class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]



# LOAD MODEL 
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

print("ResNet18 model loaded successfully!")



# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# FACE DETECTOR
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# SMOOTHING VARIABLES

ema_probs = None
current_label = "Detecting..."
candidate_label = None
candidate_count = 0


# OPEN CAMERA
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit")


# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face_pil = Image.fromarray(face_rgb)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        #EMA smoothing
        if ema_probs is None:
            ema_probs = probs
        else:
            ema_probs = EMA_ALPHA * ema_probs + (1 - EMA_ALPHA) * probs

        pred_idx = np.argmax(ema_probs)
        confidence = ema_probs[pred_idx]

        predicted_label = class_names[pred_idx]

        #Switch logic
        if confidence > CONF_THRESHOLD:

            if predicted_label == current_label:
                candidate_label = None
                candidate_count = 0

            else:
                if predicted_label == candidate_label:
                    candidate_count += 1
                else:
                    candidate_label = predicted_label
                    candidate_count = 1

                if candidate_count >= SWITCH_FRAMES:
                    current_label = candidate_label
                    candidate_label = None
                    candidate_count = 0

        text = f"{current_label} ({confidence*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Camera (Advanced Stable)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

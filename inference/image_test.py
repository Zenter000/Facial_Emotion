import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image

image_path = r"D:\Facial_Emotion\data\raw\3in1.png"
model_path = r"D:\Facial_Emotion\models\checkpoint_ResNet18\run_20260219_193106\best_epoch_2_valacc_70.86%.pth"

num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "angry","disgust","fear","happy",
    "neutral","sad","surprise"
]

print("Using device:", device)

#LOAD MODEL

model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

print("Model loaded!")

#TRANSFORM

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#FACE DETECTOR

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#LOAD IMAGE

image = cv2.imread(image_path)
if image is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#REDICT

for (x, y, w, h) in faces:

    face = image[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face_pil = Image.fromarray(face_rgb)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(face_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    label = class_names[pred_idx]

    text = f"{label} ({confidence*100:.1f}%)"

    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(image, text, (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2)

cv2.imshow("Image Emotion Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

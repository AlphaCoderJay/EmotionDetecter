import cv2
import torch
import numpy as np
import argparse
import torch.nn.functional as nnf

from torchvision import transforms
from model import Resnet18
from facenet_pytorch import MTCNN

# -------------------- ARGPARSE --------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--video", type=str, default=0,
                    help="video path or 0 for webcam")
parser.add_argument("-c", "--confidence", type=float, default=0.9,
                    help="face detection confidence")
args = vars(parser.parse_args())

device = "cpu"

# -------------------- EMOTION LABELS --------------------
emotion_dict = {
    0: "Angry",
    1: "Fearful",
    2: "Happy",
    3: "Neutral",
    4: "Sad",
    5: "Surprised"
}

# -------------------- LOAD MODEL --------------------
emotion_model = Resnet18(img_channels=1, num_classes=len(emotion_dict))
checkpoint = torch.load("emotiondetecter_model.pth", map_location=device)
emotion_model.load_state_dict(checkpoint["model_state_dict"])
emotion_model.eval()

# -------------------- MTCNN (FAST + NO TF) --------------------
mtcnn = MTCNN(keep_all=True, device=device)

# -------------------- TRANSFORM --------------------
transform_test = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# -------------------- VIDEO --------------------
vs = cv2.VideoCapture(args["video"])

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break

    output = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb)

    canvas = np.zeros((300, 300, 3), dtype="uint8")

    if boxes is not None:
        for box, prob in zip(boxes, probs):

            if prob < args["confidence"]:
                continue

            x1, y1, x2, y2 = box.astype(int)

            # clamp
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = rgb[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # -------------------- PREPROCESS --------------------
            face = transform_test(face).unsqueeze(0)

            # -------------------- PREDICT --------------------
            with torch.no_grad():
                out = emotion_model(face)
                probas = nnf.softmax(out, dim=1)

            top_p, top_class = probas.topk(1)
            top_p, top_class = top_p.item(), top_class.item()

            # -------------------- DRAW BOX --------------------
            label = f"{emotion_dict[top_class]}: {top_p*100:.2f}%"

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # -------------------- PROB DISPLAY --------------------
            for i, (emo, p) in enumerate(zip(emotion_dict.values(), probas[0])):
                text = f"{emo}: {p.item()*100:.1f}%"
                width = int(p.item() * 300)

                cv2.rectangle(canvas, (5, i*50+5), (width, i*50+50),
                              (0, 0, 255), -1)

                cv2.putText(canvas, text, (5, i*50+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

    cv2.imshow("Emotion Detection", output)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()

import torch
import torch.nn as nn
import json
from torchvision import transforms
from PIL import Image

class DeepCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ====================== LOAD MODEL ======================
device = "cpu"

model = DeepCNN(num_classes=4)   # هيتعدل تلقائياً حسب classes.json
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

with open("classes.json") as f:
    classes = json.load(f)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)[0]

    pred = torch.argmax(probs).item()

    return {
        "prediction": classes[pred],
        "confidence": float(probs[pred]),
        "all_probs": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }


# اختبار سريع
if __name__ == "__main__":
    print("Model loaded successfully!")
    print(f"Number of classes: {len(classes)}")
    print("Classes:", classes)

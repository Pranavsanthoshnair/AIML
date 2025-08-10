from deepface import DeepFace
from PIL import Image
image_path = "/s5.jpg"
result = DeepFace.analyze(image_path, actions = ["emotion"])
print("Detected Emotion:", result[0]["dominant_emotion"])
Image.open(image_path).show()

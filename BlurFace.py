import cv2                                                # cv - computer vision from open cv 
from google.colab.patches import cv2_imshow

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load image
image = cv2.imread("demo4.jpeg")                          # name of the source file in ()

# Detect faces
faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=7)

# Blur each face
for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]
    face = cv2.GaussianBlur(face, (99, 99), 30)           # 99 % BLUR, 30 - Spread
    image[y:y+h, x:x+w] = face                            # y - y-axis, x - x-axis, w - width, h - height

# Show result
cv2_imshow(image)

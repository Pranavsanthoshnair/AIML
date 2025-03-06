import cv2
from google.colab.patches import cv2_imshow
image = cv2.imread("/s3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray)
blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
sketch = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)
cv2_imshow(sketch)

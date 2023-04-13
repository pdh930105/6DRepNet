# Import SixDRepNet
from sixdrepnet import SixDRepNet
import cv2

# Create model
# Weights are automatically downloaded
model = SixDRepNet()

img = cv2.imread('./image.jpg')

pitch, yaw, roll = model.predict(img)

model.draw_axis(img, yaw, pitch, roll)

cv2.imwrite("test_image.jpg", img)


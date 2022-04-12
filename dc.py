import cv2
from SimpleHigherHRNet import SimpleHigherHRNet

model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
image = cv2.imread("image.jpg", cv2.IMREAD_COLOR)

joints = model.predict(image)
print(joints)


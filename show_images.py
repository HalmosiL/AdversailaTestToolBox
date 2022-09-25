import glob
import cv2

normal = glob.glob("./images/*")
normal.sort()

target = glob.glob("./target/*")
target.sort()

for i in range(len(target)):
    img = cv2.imread(f"./images/image_{i}_.png", 1)
    tar = cv2.imread(f"./target/image_{i}_.png", 1)
    cv2.imshow('image',img)
    cv2.imshow('taget',tar)
    cv2.waitKey(0)
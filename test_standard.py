import sys
import torch
import numpy as np
import torchvision
import pyignite as ignite
sys.path.insert(1, './')

from torchmetrics import JaccardIndex

from Dataset import SemData
from Model import load_model
from Meatrics import intersectionAndUnion
from AverageMeter import AverageMeter
import Transforms as transform
import cv2

model = load_model("./Models/Adversarial_version02_onlyAdversarialImages_cosineInner_step3_e0.03_SGD_pretrainedNormalResnet_PSNET_AUX_batch16__270.pt", "cuda:0").eval()

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

dataset = SemData(
    split='val',
    data_root="../CitySpace/Data/ok/",
    data_list="./val_list.txt",
    transform=transform.Compose([
            transform.Crop([449, 449], crop_type='center', padding=mean, ignore_label=255),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
)

intersection_meter = AverageMeter()
union_meter = AverageMeter()
target_meter = AverageMeter()

for i in range(dataset.__len__()):
    image, target = dataset.__getitem__(i)
    image = image.to("cuda:0")
    target = target.to("cuda:0")

    image = image.reshape((1, *image.shape))
    target = target.reshape((1, *target.shape))

    pred, _ = model(image)
    save_image = pred.detach().cpu().numpy().reshape((449, 449, 1)) * int(255/20)
    cv2.imwrite(f"./images/image_{i}_.png", save_image)
    save_image = cv2.imread(f"./images/image_{i}_.png", 0)
    save_image = cv2.applyColorMap(save_image, cv2.COLORMAP_JET)
    cv2.imwrite(f"./images/image_{i}_.png", save_image)

    save_image = target.detach().cpu().numpy().reshape((449, 449, 1)) * int(255/20)
    cv2.imwrite(f"./target/image_{i}_.png", save_image)
    save_image = cv2.imread(f"./target/image_{i}_.png", 0)
    save_image = cv2.applyColorMap(save_image, cv2.COLORMAP_JET)
    cv2.imwrite(f"./target/image_{i}_.png", save_image)

    intersection, union, target = intersectionAndUnion(pred, target, 19, 255)
    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    print("Finished:", i * 100/dataset.__len__(), "%")

mIoU = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

print("Acc_All:", allAcc)
print("mIOU:", mIoU)


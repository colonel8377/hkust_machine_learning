import os

import numpy
import torch
from cv2 import cv2
from torchvision import transforms

from models import DLA
from start import prepare_device, build_model, IMAGE_SIZE


def predict_image(predict_path, model, _device):
    assert os.path.isdir(predict_path)
    image_datas = prepare_prediction(predict_path + '/', os.listdir(predict_path))
    predict(model, device, image_datas)


def prepare_prediction(base, filenames):
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    images = []
    for filename in filenames:
        if not filename.endswith('jpg') and not filename.endswith('jpeg') and not filename.endswith('png'):
            continue
        image = cv2.imdecode(numpy.fromfile(base + filename, dtype=numpy.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (height, width))
        transform_pred = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        image = transform_pred(image).unsqueeze(0)
        images.append(image)
    return images


# predict function
def predict(model, device, image_datas):
    with torch.no_grad():
        for image_data in image_datas:
            inputs = image_data.to(device)
            outputs = model(inputs)
            score, predicted = outputs.max(1)
            print(score)
            print(predicted)


if __name__ == '__main__':
    device = prepare_device()
    net, device_nums = build_model(device, DLA())
    print('==> predicting from checkpoint..')
    assert os.path.isdir('outputs'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./outputs/ckpt.pth', map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    predict_image('predict', net, device)

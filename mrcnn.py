from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F  #GG
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights   #GG
import PIL  #GG
import torchvision
import torch
import numpy as np
import cv2
import random
import time
import os

# if torch.cuda.is_available():  
#   device = torch.device("cuda:0")
# else:  
#   device = torch.device("cpu")


# print("Device:", device)

# These are the classes that are available in the COCO-Dataset
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# # get the pretrained model from torchvision.models
# # Adding "device" to use GPU
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model.to(device)
# model.eval()

def random_colour_masks(image):
    """
    random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, threshold):
    """
    get_prediction
    parameters:
      - img_path - path of the input image
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
        ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def instance_segmentation_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    """
    instance_segmentation_api
    parameters:
      - img_path - path to input image
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# We will use the following colors to fill the pixels
colours = [[0, 255, 0],
           [0, 0, 255],
           [255, 0, 0],
           [0, 255, 255],
           [255, 255, 0],
           [255, 0, 255],
           [80, 70, 180],
           [250, 80, 190],
           [245, 145, 50],
           [70, 150, 250],
           [50, 190, 190]]


def main(image_path):
  if torch.cuda.is_available():  
    device = torch.device("cuda:0")
  else:  
    device = torch.device("cpu")

  print("Device:", device)

  # get the pretrained model from torchvision.models
  # Adding "device" to use GPU
  model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
  model.to(device)
  model.eval()

  img = Image.open(image_path)
  print("Image size: ", img.size)
  print("Image format: ", img.mode)

  img_tensor = F.pil_to_tensor(img)
  img_tensor = img_tensor.to(device)  # Move image tensor to the same device as the model
  preprocess_img = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
  # preprocess_img = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.transforms().to(device)
  # preprocess_img = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
  # pred = model(preprocess_img(img_tensor).unsqueeze(dim=0).to(device))
  pred = model(preprocess_img(img_tensor).unsqueeze(dim=0))

  # Lets look at what the `pred` looks like.
  # `pred` is a list of dictionaries, since we had passed a single image, we will get a single-item list
  # print("pred:", pred[0])

  # We will keep only the pixels with values  greater than 0.5 as 1, and set the rest to 0.
  if pred[0]['masks'].shape[0] == 1:
     masks = pred[0]['masks'][:][0]
     masks = (masks>0.5).detach() 
  else:
    masks = (pred[0]['masks']>0.5).squeeze().detach() 

  # Let's plot the mask for the `person` class since the 0th mask belongs to `person`
  
  masks = masks.cpu().numpy()  # Move masks tensor to CPU and convert to NumPy array
  plt.imshow(masks[0], cmap='gray')
  plt.show()

  # Let's color the `person` mask using the `random_colour_masks` function
  mask1 = random_colour_masks(masks[0])

  # Let's blend the original and the masked image and plot it.
  blend_img = cv2.addWeighted(np.asarray(img), 0.5, mask1, 0.5, 0)

  plt.imshow(blend_img)
  plt.show()

if __name__=='__main__':
  # path = 'gina_2016.jpg'    # works
  path = './gina_oct2018.jpg'   # works with my fix
  # path = './MAX_19-23 Surface white background.tif'  #doesn't work
  # path = 'mrcnn_standing_people.jpg'    # works, one instance
  main(path)
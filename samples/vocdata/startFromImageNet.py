#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
-----------------------------------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a model from coco weights.
    python3 voc.py train --dataset=/path/to/VOCdevkit/ --model=coco --year=2012
    # Train a new model starting from ImageNet weights.
    python3 voc.py train --dataset=/path/to/VOCdevkit/ --model=imagenet --year=2012
    # Continue training a model that you had trained earlier
    python3 voc.py train --dataset=/path/to/VOCdevkit/ --model=/path/to/weights.h5  --year=2012
    # Continue training the last model you trained
    python3 voc.py train --dataset=/path/to/VOCdevkit/ --model=last
    # Run VOC inference on the last model you trained
    python3 voc.py inference --dataset=/path/to/VOCdevkit/ --model=last --year=2012 --limit=50
------------------------------------------------------------------------------------------    
"""


# In[5]:


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import imgaug


# In[6]:


import matplotlib.pyplot as plt
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# In[26]:


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Inference result directory
RESULTS_DIR = os.path.abspath("./inference/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# %matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR =  MODEL_DIR
DEFAULT_DATASET_YEAR = ''
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# In[16]:


# VOC DATASET MASK MAP FUNCTION
# Following codes are mapping each mask color(SegmentationClass) to ground truth index.
# - reference: https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tvmonitor']


# In[17]:


def build_colormap2label():
    """Build a RGB color to label mapping for segmentation."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i
    return colormap2label


# In[18]:


def voc_label_indices(colormap, colormap2label):
    """Map a RGB color to a label."""
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
# VOC DATASET MASK MAP FUNCTION


# In[19]:


class VocConfig(Config):
    NAME = "voc"

    IMAGE_PER_GPU = 2

    NUM_CLASSES = 1 + 20 # VOC 2012 have 20 classes. "1" is for background.


# In[20]:


class InferenceConfig(VocConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0


# In[21]:


#voc = COCO("pascal_train.json") # load training annotations


# In[22]:


class VocDataset(utils.Dataset):
    def load_voc(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
              class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco = COCO("../vocdata/pascal_train.json")
       
        image_dir = dataset_dir

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco
    def load_voc000(self, dataset_dir, trainval, year=''):
        """Load a voc_year of the VOC dataset.
        dataset_dir: The root directory of the VOC dataset, example: '/samples/vocdata/'
        trainval: 'train' or 'val' for Training or Validation
        year: all as empty string ''
        """

        voc_year = year
        Segmentation = os.path.join(dataset_dir, voc_year, 'ImageSets', 'Segmentation')
        JPEGImages = os.path.join(dataset_dir, voc_year, 'JPEGImages')
        Annotations = os.path.join(dataset_dir, voc_year, 'Annotations')
        SegmentationClass = os.path.join(dataset_dir, voc_year, 'SegmentationClass')
        SegmentationObject = os.path.join(dataset_dir, voc_year, 'SegmentationObject')

        # load classes of VOC, BG is initialed in parent class.
        for idx, class_name in enumerate(VOC_CLASSES[1:]):
            self.add_class("voc", idx + 1, class_name)

        assert trainval in ['train', 'val']
        # read segmentation annotation file
        annotation_file = os.path.join(Segmentation, trainval + '.txt')
        image_ids = []
        with open(annotation_file) as f:
            image_id_list = [line.strip() for line in f]
            image_ids += image_id_list

        for image_id in image_ids:
            image_file_name = '{}.jpg'.format(image_id)
            mask_file_name = '{}.png'.format(image_id)
            xml_file_name = '{}.xml'.format(image_id)
            image_path = os.path.join(JPEGImages, image_file_name)

            # Parse Annotations XML File
            with open(os.path.join(Annotations, xml_file_name)) as f:
                soup = bs(f, 'xml')
            objects = soup.find_all('object')
            image_contains_class_flag = False
            for obj in objects:
                class_name = obj.find('name').text
                if class_name in VOC_CLASSES:
                    image_contains_class_flag = True
                    continue
            if image_contains_class_flag:
                class_mask_path = os.path.join(SegmentationClass, mask_file_name)
                object_mask_path = os.path.join(SegmentationObject, mask_file_name)
                self.add_image("voc",
                                image_id=image_file_name,
                                path=image_path,
                                class_mask_path=class_mask_path,
                                object_mask_path=object_mask_path)
    def load_raw_mask(self, image_id, class_or_object):
            '''load two kinds of mask of VOC dataset.
            image_id: id of mask
            class_or_object: 'class_mask' or 'object_mask' for SegmentationClass or SegmentationObject
            Returns:
            image: numpy of mask image.
            '''
            image = skimage.io.imread(self.image_info[image_id][class_or_object+'_path'])
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
            return image

    def load_class_label(self, image_id):
        '''Mapping SegmentationClass image's color to indice of ground truth 
        image_id: id of mask
        Return:
        class_label: [height, width] matrix contains values form 0 to 20
        '''
        raw_mask = self.load_raw_mask(image_id, 'class_mask')
        class_label = voc_label_indices(raw_mask, build_colormap2label())
        return class_label

    def load_mask(self, image_id):
        '''Mapping annotation images to real Masks(MRCNN needed)
        image_id: id of mask
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        '''
        class_label = self.load_class_label(image_id)
        instance_mask = self.load_raw_mask(image_id, 'object_mask')
        max_indice = int(np.max(class_label))

        instance_label = []
        instance_class = []
        for i in range(1, max_indice+1):
            if not np.any(class_label==i):
                continue
            gt_indice = i
            object_filter = class_label == i
            object_filter = object_filter.astype(np.uint8)
            object_filter = np.dstack((object_filter,object_filter,object_filter))
            filtered = np.multiply(object_filter, instance_mask)
            gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
            max_gray = np.max(gray)
            for sub_index in range(1, max_gray+1):
                if not np.any(gray==sub_index):
                    continue
                instance_filter = gray == sub_index
                instance_label += [instance_filter]
                instance_class += [gt_indice]
        masks = np.asarray(instance_label).transpose((1,2,0))
        classes_ids = np.asarray(instance_class)
        return masks, classes_ids


# In[23]:


############################################################
#  Inference
############################################################


# In[24]:


def inference(model, dataset, limit):
    """Run detection on images in the given directory."""

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    time_dir = "{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    time_dir = os.path.join(RESULTS_DIR, time_dir)
    os.makedirs(time_dir)

    # Load over images
    for image_id in dataset.image_ids[:limit]:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        # Save image with masks
        if len(r['class_ids']) > 0:
            print('[*] {}th image has {} instance(s).'.format(image_id, len(r['class_ids'])))
            visualize.display_instances(
                image, r['rois'], r['masks'], r['class_ids'],
                dataset.class_names, r['scores'],
                show_bbox=True, show_mask=True,
                title="Predictions")
            plt.savefig("{}/{}".format(time_dir, dataset.image_info[image_id]["id"]))
            plt.close()
        else:
            plt.imshow(image)
            plt.savefig("{}/noinstance_{}".format(time_dir, dataset.image_info[image_id]["id"]))
            print('[*] {}th image have no instance.'.format(image_id))
            plt.close()


# In[27]:


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on PASCAL VOC.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference' on PASCAL VOC")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/voc/",
                        help='Directory of the PASCAL VOC dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the PASCAL VOC dataset (2007 or 2012) (default=2012)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'voc'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=10,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=10)')
    # TODO
    '''
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip PASCAL VOC files (default=False)',
                        type=bool)
    '''
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    #print("Auto Download: ", args.download)


    # Configurations
    if args.command == "train":
        config = VocConfig()
    else:
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_WEIGHTS_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)


    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = VocDataset()
        dataset_train.load_voc(args.dataset, "train", year=args.year)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = VocDataset()
        dataset_val.load_voc(args.dataset, "val", year=args.year)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "inference":
        #print("evaluate have not been implemented")
        # Validation dataset
        dataset_val = VocDataset()
        voc = dataset_val.load_voc(args.dataset, "val", year=args.year)
        dataset_val.prepare()
        print("Running voc inference on {} images.".format(args.limit))
        inference(model, dataset_val, int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





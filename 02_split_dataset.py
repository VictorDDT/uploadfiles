#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, os, random, math, shutil, cv2
import tensorflow as tf
import xml.etree.ElementTree as ET

from classes.config_class import Config
from classes.utils_class import Utils

C = Config()
U = Utils()

fileList = []
count = 0
for file in glob.glob(os.path.join(C.get('imagePathAugmented'), '*')):
    if os.path.isfile(file):
        count += 1
        fileID = os.path.basename(file).split('.')[0]
        fileList.append(fileID)

random.shuffle(fileList)

if C.get('imageCountLimit') is not None:
    totalCount = C.get('imageCountLimit')
else:
    totalCount = len(fileList)
trainLength = math.ceil(C.get('trainSize') * totalCount)

# clean Train and Eval folders
U.clearFolder(C.get('imagePathAugmentedTrain'))
U.clearFolder(C.get('annotationPathAugmentedTrain'))
U.clearFolder(C.get('imagePathAugmentedEval'))
U.clearFolder(C.get('annotationPathAugmentedEval'))

if os.path.exists(C.get('recordPathAugmentedTrain')):
    os.remove(C.get('recordPathAugmentedTrain'))

if os.path.exists(C.get('recordPathAugmentedEval')):
    os.remove(C.get('recordPathAugmentedEval'))


# copy Train and Eval files from augmented dataset
tfWriterTrain = tf.python_io.TFRecordWriter(C.get('recordPathAugmentedTrain'))
tfWriterEval = tf.python_io.TFRecordWriter(C.get('recordPathAugmentedEval'))

count = 0
for fileID in fileList:
    count += 1
    print("image: {0:,}".format(count))

    imagePathAugmented = os.path.join(C.get('imagePathAugmented'), fileID + '.jpg')
    annotationPathAugmented = os.path.join(C.get('annotationPathAugmented'), fileID + '.xml')

    # Calculate target path (train or eval folder) based on image count
    if count < trainLength:
        imagePathTarget = os.path.join(C.get('imagePathAugmentedTrain'), fileID + '.jpg')
        annotationPathTarget = os.path.join(C.get('annotationPathAugmentedTrain'), fileID + '.xml')
    else:
        imagePathTarget = os.path.join(C.get('imagePathAugmentedEval'), fileID + '.jpg')
        annotationPathTarget = os.path.join(C.get('annotationPathAugmentedEval'), fileID + '.xml')

    # Read annotation XML file
    xmlData = ET.parse(annotationPathAugmented)
    root = xmlData.getroot()
    imageWidth = int(root.find('size').find('width').text)
    imageHeight = int(root.find('size').find('height').text)

    # Check augmented image real size (width and height)
    # If real size is not equal to "C.imageSize" than resize the image and update XML annotation
    imageAug = cv2.imread(imagePathAugmented)
    if imageAug.shape[0] != C.get('imageSize') or imageAug.shape[1] != C.get('imageSize'):
        # Resize and save new image to the target folder
        imageWidth = imageHeight = C.get('imageSize')
        imageAug = cv2.resize(imageAug, (imageWidth, imageHeight))
        cv2.imwrite(imagePathTarget, imageAug)

        # Update and save annotation XML file
        root.find('size').find('width').text = str(imageWidth)
        root.find('size').find('height').text = str(imageHeight)
        xmlData.write(annotationPathTarget)
    else:
        # Image size is equal to "C.imageSize". Just copy files to the target folder. No changes.
        shutil.copyfile(imagePathAugmented, imagePathTarget)
        shutil.copyfile(annotationPathAugmented, annotationPathTarget)

    # Preare features and create TFRecord object
    with tf.gfile.GFile(imagePathTarget, 'rb') as fid:
        encoded_jpg = fid.read()

    xmin = float(int(root.find('object').find('bndbox').find('xmin').text) / imageWidth)
    ymin = float(int(root.find('object').find('bndbox').find('ymin').text) / imageHeight)
    xmax = float(int(root.find('object').find('bndbox').find('xmax').text) / imageWidth)
    ymax = float(int(root.find('object').find('bndbox').find('ymax').text) / imageHeight)

    features = {}
    features['imageHeight'] = imageHeight
    features['imageWidth'] = imageWidth
    features['imagePathAugmented'] = imagePathAugmented
    features['encoded_jpg'] = encoded_jpg
    features['xmin'] = xmin
    features['xmax'] = xmax
    features['ymin'] = ymin
    features['ymax'] = ymax
    features['classes_text'] = 'knuckles'
    features['classes'] = 1
    tfRecord = U.createTFRecord(features)

    # Send TFRecord object to train / eval TFWriter
    if count < trainLength:
        tfWriterTrain.write(tfRecord.SerializeToString())
    else:
        tfWriterEval.write(tfRecord.SerializeToString())

    if C.get('imageCountLimit') is not None and count == C.get('imageCountLimit'):
        break

tfWriterTrain.close()
tfWriterEval.close()

print("=============")
print("*** DONE ***")
print("=============")

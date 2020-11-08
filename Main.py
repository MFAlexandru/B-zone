#This is just a demo and should not be evaluated as a final
#product

#This scrips calls the image recogition AI on a specific sample

import cv2

import numpy as np

# Import footage
cap = cv2.VideoCapture('Sample.mp4')
# Set AI Bound
whT = 608
confThreshold = 0.5
nmsThreshold = 0.3

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1280, 720))

# Import AI model
classFile = 'coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3-608.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Recognise and mark objects on picture
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox,confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w,y + h), (255, 0 ,255), 2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0 ,255), 2)

# Resize the image for better performance
def resize(img):
    resized_frame = cv2.resize(img, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    return resized_frame

# Use the model on an image
def recognise(img):
    blob = cv2.dnn.blobFromImage(img , 1 / 255, (whT, whT),[0,0,0],1, crop = False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    findObjects(outputs, img)

# Analyse The video
while True:
    ret, frame = cap.read()
    if ret == 0:
        break

    resized_frame = resize(frame)

    recognise(resized_frame)
    out.write(resized_frame)
    cv2.imshow('Image', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

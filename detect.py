import cv2
import numpy as np


def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.4:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    # print(len(boxes))
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    count = 0
    for i in range(len(boxes)):
        if i in indexes and class_ids[i]==0:
            count +=1
            x, y, w, h = boxes[i]
            color = [0,0,255]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            # cv2.putText(img,label, (x, y - 5), font, 2, color, 1)
    return img,count

def load_image(gambar):
	# image loading
    img = gambar
    height, width, channels = img.shape
    return img, height, width, channels

def image_detect(img_path,model,classes,colors,output_layers): 
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    gambar,count = draw_labels(boxes, confs, colors, class_ids, classes, image)
    return gambar,count

model, classes, colors, output_layers = load_yolo()
cap = cv2.VideoCapture(0)
while True:
    _,im = cap.read()
    imageProcess,jumlahOrang = image_detect(im,model,classes,colors,output_layers)
    color = [0,0,0]
    cv2.rectangle(imageProcess, (10,420), (320,460), (255,255,255), -1)
    cv2.putText(imageProcess,"Jumlah Orang: {}".format(jumlahOrang), (20,450), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
    cv2.imshow('detection',imageProcess)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()


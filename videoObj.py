import cv2
import os
import numpy as np

print("Available devices are:")
print("Device list: ", os.system("ls /dev/video*"))
answer = input("Choose a number from the device array: ")

# Load the network
net = cv2.dnn.readNetFromDarknet("/home/ajey/Desktop/Rayreach/Live-OD-tool/yolov3.cfg", "/home/ajey/Desktop/Rayreach/Live-OD-tool/yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

names = "/home/ajey/Desktop/Rayreach/darknet/data/coco.names"

CONF_THRESH, NMS_THRESH = 0.5, 0.5

layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cam = cv2.VideoCapture(int(answer))
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
count = 0
while True:
    ## read frames
    ret, img = cam.read()
    count = count + 1
    if img is not None and count%2==0:
        height, width = img.shape[:2]
        ## predict yolo
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        class_ids, confidences, b_boxes = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH)
        if len(indices) != 0:
            indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
        # Draw the filtered bounding boxes with their class to the image
        with open(names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        for index in indices:
            # print("boxes present!")
            x, y, w, h = b_boxes[index]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0), 3)
            cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)
        ## display predictions
        cv2.imshow("Live Feed", img)
        ## press q or Esc to quit
        if cv2.waitKey(33) & 0xff == 27:
            cv2.destroyAllWindows()
            cam.release()
            break
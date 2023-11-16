import numpy as np
import cv2 as cv

net = cv.dnn.readNet('darknet/yolov3.weights', 'darknet/cfg/yolov3.cfg')

classes = []
with open('darknet/data/coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not openning!!")
    exit()

while True:
    ret, frame = cap.read()

    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    #  process detection results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv.putText(frame, classes[class_id], (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv.imshow('Object Detection', frame)
    if cv.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

    if not ret:
        print("Can't receive frame! Exiting")
        break
    
    # background = cv.cvtColor(frame, cv.COLOR_BGR)
    # cv.imshow('frame', background)
    if cv.waitKey(1) ==  ord('q'):
        break

# when everything done, release the capture
cap.release()
cv.destroyAllWindows()

import tkinter as tk
import tkinter.font as font
import tkinter.ttk as ttk
from tkinter import *
import tensorflow as tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from scipy.spatial import distance as dist

cap = cv2.VideoCapture(0)
face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

window = tk.Tk()  
window.title("Face mask and social distancing")
window.geometry('1920x1080')
window.configure(background ='white') 
window.grid_rowconfigure(0, weight = 1) 
window.grid_columnconfigure(0, weight = 1) 
message = tk.Label( 
    window, text ="Face Mask & social distance detection",  
    bg ="lightsalmon", fg = "black", width = 50,  
    height = 3, font = ('times', 30, 'bold'))

message.place(x = 100, y = 20)


def face_mask():
    
    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # ensure the bounding boxes fall within the dimensions of
                        # the frame
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                        # extract the face ROI, convert it from BGR to RGB channel
                        # ordering, resize it to 224x224, and preprocess it
                        face = frame[startY:endY, startX:endX]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = cv2.resize(face, (224, 224))
                        face = img_to_array(face)
                        face = preprocess_input(face)
                        face = np.expand_dims(face, axis=0)

                        # add the face and bounding boxes to their respective
                        # lists
                        faces.append(face)
                        locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                preds = maskNet.predict(faces)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

def weapon_detection():
	os.system('python weapon.py')


def social_distancing():
    while True:
        status , photo = cap.read()
        face_cor = face_model.detectMultiScale(photo)
        l = len(face_cor)
        photo = cv2.putText(photo, str(len(face_cor))+" Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        stack_x = []
        stack_y = []
        stack_x_print = []
        stack_y_print = []
        global D

        if len(face_cor) == 0:
            pass
        else:
            for i in range(0,len(face_cor)):
                x1 = face_cor[i][0]
                y1 = face_cor[i][1]
                x2 = face_cor[i][0] + face_cor[i][2]
                y2 = face_cor[i][1] + face_cor[i][3]

                mid_x = int((x1+x2)/2)
                mid_y = int((y1+y2)/2)
                stack_x.append(mid_x)
                stack_y.append(mid_y)
                stack_x_print.append(mid_x)
                stack_y_print.append(mid_y)

                photo = cv2.circle(photo, (mid_x, mid_y), 3 , [255,0,0] , -1)
                photo = cv2.rectangle(photo , (x1, y1) , (x2,y2) , [0,255,0] , 2)

            if len(face_cor) == 2:
                D = int(dist.euclidean((stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())))
                photo = cv2.line(photo, (stack_x_print.pop(), stack_y_print.pop()), (stack_x_print.pop(), stack_y_print.pop()), [0,0,255], 2)
            else:
                D = 0

            if D<250 and D!=0:
                photo = cv2.putText(photo, "PLEASE MAINTAIN SOCIAL DISTANCING!!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,1, [0,0,255] , 2)

            photo = cv2.putText(photo, str(D/10) + " cm", (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0) , 2, cv2.LINE_AA)

            cv2.imshow('hi' , photo)
            if cv2.waitKey(100) == 13:
                break

    cv2.destroyAllWindows()

   

takeImg = tk.Button(window, text ="Face Mask",  
command = face_mask, fg ="black", bg ="orange",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
takeImg.place(x = 100, y = 400)

trainImg = tk.Button(window, text ="Social Distance",  
command = social_distancing, fg ="black", bg ="lightgreen",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trainImg.place(x = 500, y = 400)

#trainImg = tk.Button(window, text ="Weapon Detection",  
#command = weapon_detection, fg ="black", bg ="lightblue",  
#width = 20, height = 3, activebackground = "Red",  
#font =('times', 15, ' bold ')) 
#trainImg.place(x = 900, y = 400)


window.mainloop() 


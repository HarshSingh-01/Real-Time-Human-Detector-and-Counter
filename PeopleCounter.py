from myUtils.centroidtracker import CentroidTracker
from myUtils.trackableobject import TrackableObject
from myUtils.mailer import Mailer

from imutils.video import VideoStream
from imutils.video import FPS
from myUtils import config, thread

import time
import csv
import schedule
import numpy as np
import argparse
import imutils
import dlib
import cv2
import datetime
from itertools import zip_longest

t0 = time.time()


def run():
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    # Cofidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())

    # Intialize the list of class labels MobileNet SSD was trained to detect.
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Load our serialized model from disk.
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # If a video path was not supplied, grab a refrence to the ip camera
    if not args.get("input", False):
        print("[INFO] Starting the Video file url..")
        vs = VideoStream(config.url).start()
        time.sleep(2.0)
    # Starting the webcam if IP camera is not present
    elif not args.get("input", False):
        print("[INFO] Starting the Webcam..")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    # Otherwise, grab a refrence to the video file.
    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])

    # Intialize the video writer (we'll instaniate later if need be)
    writer = None

    # Initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    H = None
    W = None

    # Initialize our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject.
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # Initializa the total number of frames processed thus far, along
    # with the total number of objects that have moved either Left or Right.
    totalFrames = 0
    totalLeft = 0
    totalRight = 0
    x = []
    empty = []
    empty1 = []

    # Start the frames per second throughout estimator
    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    # loop over frames from the video stream.
    while True:
        # Grab the next frame and handle if we are reading from either VideCapture or VideoStream.
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        if args["input"] is not None and frame is None:
            break

        # Resize the frame to have a maximum width of 500 pixels (the less data
        # we have, the faster we can process it), then convert
        # the frame from RGB for dlib.
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # If the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize the writer.
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30.0, (W, H))
            # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        # Inintialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers.
        status = "Waiting"
        rects = []

        # Check to see if we should run a more coputationally expensive
        # object detection method to aid our tracker.
        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []

            # Convert the frame to a blob and pass the blob through the network
            # and obtain the detectons.
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)

        else:
            for tracker in trackers:
                status = "Tracking"

                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (W//2, 0), (W//2, H), (255, 255, 255), 3)
        # cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i*20)+200)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H//2:
                        totalRight += 1
                        empty.append(totalRight)
                        to.counted = True

                    elif direction > 0 and centroid[1] > H//2:
                        totalLeft += 1
                        empty1.append(totalLeft)
                        x = []
                        x.append(len(empty1)+len(empty))

                        if sum(x) >= config.Threshold:
                            cv2.putText(frame, "-Alert: People limit exceeded-", (10,
                                                                                  frame.shape[1] - 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            if config.ALERT:
                                print("[INFO] Sending email alert..")
                                Mailer().send(config.MAIL)
                                print("[INFO] Alert sent")

                        to.counted = True

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)

            cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(
                frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        info = [("Right", totalRight), ("Left", totalLeft), ("Status", status)]
        info2 = [("T. people inside", x)]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H-((i*20) + 20)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (W//2+10, H-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

        if config.Log:
            dateTime = [datetime.datetime.now()]
            d = [dateTime, empty1, empty, x]
            export_data = zip_longest(*d, fillvalue='')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow("End Time", "In", "Out", "Total Inside")
                wr.writerows(export_data)

        cv2.imshow("Analyis Window", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

        if config.Timer:
            t1 = time.time()
            num_seconds = (t1-t0)
            if num_seconds > 28800:
                break

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()


if config.Scheduler:
    schedule.every().day.at("9:00").do(run)

    while 1:
        schedule.run_pending()

else:
    run()

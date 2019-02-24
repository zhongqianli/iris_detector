import numpy as np
import argparse
import cv2 as cv
try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

from cv2 import dnn

inWidth = 300
inHeight = 300
confThreshold = 0.5

prototxt = 'deploy.half.prototxt'
caffemodel = 'res10_300x300_ssd.half_iter_140000.caffemodel'

if __name__ == '__main__':
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    while True:
        frame = cv.imread("../../images/S2353L09.jpg", 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cols = frame.shape[1]
        rows = frame.shape[0]

        net.setInput(dnn.blobFromImage(gray, 1.0, (inWidth, inHeight), (128), False, False))
        detections = net.forward()

        # print(detections)

        perf_stats = net.getPerfProfile()

        infer_time = perf_stats[0] / cv.getTickFrequency() * 1000
        fps = 1000 / infer_time
        fps_time_str = 'fps = {0}, time = {1} ms'.format(int(fps), int(infer_time))
        cv.putText(frame, fps_time_str, (50, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))	

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                             (0, 255, 0))
                label = "iris: %.4f" % confidence
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (0, 0, 0), cv.FILLED)
                cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        if frame.shape[1] > 800 or frame.shape[0] > 800:
            frame = cv.resize(frame, dsize=(0,0), fx=0.5, fy=0.5)
        cv.imshow("detections", frame)
        if cv.waitKey(1) == int(ord('s')):
            cv.imwrite("result.bmp", frame)

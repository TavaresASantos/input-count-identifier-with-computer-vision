import numpy as np
import cv2


captura = cv2.VideoCapture('media/1.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

while 1:
    ret, frame = captura.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    retval, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    open = cv2.morphologyEx(thresh, cv2.MORPH_ELLIPSE, kernel)
    dilation = cv2.dilate(open, kernel, iterations=8)
    clossing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Video Frame open', clossing)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
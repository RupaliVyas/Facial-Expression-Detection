import numpy as np
import cv2

cap = cv2.cv2.VideoCapture(0)
i = 0
while(True):
 
    ret, frame = cap.read()
    # gray = cv2.cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2GRAY)
 
    cv2.cv2.imshow('frame',frame)
    # name = ' ' + str(i) + '.jpg'
    # print ('Creating...' + name)
    # cv2.cv2.imwrite(name, frame)

    i+=1
    if cv2.cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.cv2.destroyAllWindows()
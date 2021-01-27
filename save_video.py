import time

import cv2
import numpy as np
from mss import mss


def record(name):
    with mss() as sct:
        # mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}
        mon = sct.monitors[0]
        name = name + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        desired_fps = 30.0
        out = cv2.VideoWriter(name, fourcc, desired_fps,
                              (mon['width'], mon['height']))
        last_time = 0
        while True:
            frame = sct.grab(mon)
            #cv2.imshow('test', np.array(frame))
            img = np.array(frame)
            hsv_frame = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            low_green = np.array([25, 52, 72])
            high_green = np.array([102, 255, 255])
            red_mask = cv2.inRange(hsv_frame,low_green,high_green)
            contours,hierachy= cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
            obj = 1
            for cnt in contours:
                try:
                    (x,y,w,h) = cv2.boundingRect(cnt)
                    if (40<= w <= 100 and 30<=h<=40):
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                        cv2.putText(img,str(obj),(x+w,y+h+3),cv2.FONT_HERSHEY_SIMPLEX,1,(40,50,60),2,cv2.LINE_AA)
                        cv2.circle(img,(int((x+x+w)/2),int((y+y+h)/2)),4,(0,0,0),4)
                        cv2.line(img,(int((x+x+w)/2),int((y+y+h)/2)),(int((x+x+w)/2),bird_y_mid),(0,255,0),2)
                        obj += 1
                    elif ((40<= w <= 80 and 40<h<=50) or (40<= h <= 100 and 40<w<=50) ):
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        bird_x_mid = int((x+x+w)/2)
                        bird_y_mid = int((y+y+h)/2)
                        cv2.circle(img,(bird_x_mid,bird_y_mid),4,(0,0,0),4)
                        cv2.line(img,(0,bird_y_mid),(4000,bird_y_mid),(0,255,0),3)

                    
                except:
                    pass

            if time.time() - last_time > 1./desired_fps:
                last_time = time.time()
                destRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                out.write(destRGB)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


record("Video")
import numpy as np 
import cv2 
  
# capture frames from a camera  
cap = cv2.VideoCapture(0); 
ret, img = cap.read(); 
rows,cols, _ = img.shape
while(1): 
    # read frames 
    ret, img = cap.read(); 
    img = cv2.flip(img,1)
    hsv_frame = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    low_red = np.array([161,155,84])
    high_red = np.array([179,255,255])
    red_mask = cv2.inRange(hsv_frame,low_red,high_red)
    contours,hierachy= cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in contours:
        try:
            (x,y,w,h) = cv2.boundingRect(cnt)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            x_medium = int((x+x+w)/2)
            y_medium = int((y+y+h)/2)
            cv2.line(img,(x_medium,0),(x_medium,4080),(0,255,0),2)
            cv2.line(img,(0,y_medium),(4080,y_medium),(0,255,0),2)
            cv2.circle(img,(x_medium,y_medium),2,(0,255,0),2)
            cv2.line(img,(int(cols/2),0),(int(cols/2),4080),(190,100,40),2)
            cv2.line(img,(0,int(rows/2)),(4080,int(rows/2)),(190,100,40),2)
            cv2.line(img,(int(cols/2),int(rows/2)),(x_medium,y_medium),(0,0,255),2)
            cv2.putText(img,'Target ('+str(x_medium)+','+str(y_medium)+')',(x_medium,y_medium),cv2.FONT_HERSHEY_SIMPLEX,1,(40,50,60),2,cv2.LINE_AA)
            cv2.putText(img,'camera ('+str(int(cols/2))+','+str(int(rows/2))+')',(int(cols/2)+5,int(rows/2)+5),cv2.FONT_HERSHEY_SIMPLEX,1,(40,50,60),2,cv2.LINE_AA)
            
            slope = -1*((y_medium-int(rows/2))/(x_medium-int(cols/2)))
            cv2.putText(img,'Slope= '+str(slope),((x_medium+int(cols/2))//2,(y_medium+int(rows/2))//2),cv2.FONT_HERSHEY_SIMPLEX,1,(40,50,60),2,cv2.LINE_AA)
            if(x_medium - (int(cols)/2))>4:
                cv2.putText(img,'LEFT ',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(40,50,60),2,cv2.LINE_AA)
            elif(x_medium - (int(cols)/2))<-4:
                cv2.putText(img,'RIGHT ',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(40,50,60),2,cv2.LINE_AA)
            
            if(y_medium - (int(rows)/2))>4:
                cv2.putText(img,'UP ',(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(40,50,60),2,cv2.LINE_AA)
            elif(y_medium - (int(rows)/2))<-4:
                cv2.putText(img,'DOWN ',(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(40,50,60),2,cv2.LINE_AA)
                                    

            break
        except:
            pass

    cv2.imshow('Frame',img)
    cv2.imshow("",red_mask)
    k = cv2.waitKey(30) & 0xff; 
    if k == 27: 
        break; 
  
cap.release(); 
cv2.destroyAllWindows(); 
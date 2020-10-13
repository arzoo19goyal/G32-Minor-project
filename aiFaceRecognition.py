#main

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path=r"dataset"
images=[]
classnames=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist[1:]:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
def finden(images):
    encodelist=[]
    for i in images:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(i)[0]
        encodelist.append(encode)
    return(encodelist)
            
def mark(name):
    with open("D:/Attendance.csv","r+") as f:
        mydatalist=f.readlines()
        #print(mydatalist)
        namelist=[]
        for line in mydatalist:
            entry=line.split(",")
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            date=now.strftime("%d/%m/%y")
            dtstring=now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtstring},{date}')
            
            
encodelk=finden(images)
print("complete") 

cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    
    facescf=face_recognition.face_locations(imgs)
    encodecf=face_recognition.face_encodings(imgs,facescf)
    
    for encodeface,faceloc in zip(encodecf,facescf):
        matches=face_recognition.compare_faces(encodelk,encodeface)
        facedis=face_recognition.face_distance(encodelk,encodeface)
        matchi=np.argmin(facedis)
        
        if matches[matchi]:
            name=classnames[matchi].upper()
            #print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            mark(name)
            
            
            
            
        
    cv2.imshow("webcam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

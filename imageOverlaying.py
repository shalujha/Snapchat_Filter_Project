import cv2
import numpy as np
import pandas as pd

face=cv2.imread(r'Before.png',-1)
#face=cv2.imread(r'Jamie_Before.jpg',-1)
specs=cv2.imread(r'glasses.png',-1)
mustache=cv2.imread(r'mustache.png',-1)
#cv2.imshow("specs",specs)
#cv2.imshow("Mustache",mustache)
#gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
eyes_cascade=cv2.CascadeClassifier('frontalEyes35x16.xml')
eyes=eyes_cascade.detectMultiScale(face,1.3,5)
nose_cascade=cv2.CascadeClassifier('Nose18x15.xml')
noses=nose_cascade.detectMultiScale(face,1.3,5)
#for (x,y,w,h) in eyes:
#	cv2.rectangle(face,(x,y),(x+h,y+w),(255,67,89),2)
#for (nx,ny,nw,nh) in noses:
#	cv2.rectangle(face,(nx,ny),(nx+nh,ny+nw),(255,67,89),2)
[(x,y,w,h)]=eyes
[(nx,ny,nw,nh)]=noses
specs=cv2.resize(specs,(h+h//2,w+w//2))
mustache=cv2.resize(mustache,(h-h//3,w))
#noses=np.array(noses,dtype='uint8')
#noses=cv2.resize(noses,(w,h))
face=cv2.cvtColor(face,cv2.COLOR_BGR2BGRA)
specs=cv2.cvtColor(specs,cv2.COLOR_BGR2BGRA)
#noses=cv2.cvtColor(noses,cv2.COLOR_BGR2BGRA)
mustache=cv2.cvtColor(mustache,cv2.COLOR_BGR2BGRA)
#print(type(face))
#print(face.shape)
w,h,c=specs.shape
nw,nh,nc=mustache.shape
#print(noses.shape)
#print(mustache.shape)
#print(nw,nh,nc)
#face[y:y+w,x:x+h]=specs # default way of image overlaying
for i in range(0,w):
	for j in range(0,h):
		if specs[i][j][3]!=0:
			#face[y+i][x+j]=specs[i][j]
			face[y+i][x+j-nh//2]=specs[i][j]
for i in range(0,nw):
	for j in range(0,nh):
		if mustache[i][j][3]!=0:
			#face[ny+i+nh//4][nx+j-(nw//2)+nh//7]=mustache[i][j]
			face[ny+i+nh//3][nx+j-nw//5]=mustache[i][j]
#cv2.imshow("specs",specs)
#cv2.imshow("Face",face)

face=cv2.cvtColor(face,cv2.COLOR_BGRA2BGR)
red=[]
green=[]
blue=[]
# testing 
for i in range(face.shape[0]):
	red.append(face[i,0])
	green.append(face[i,1])
	blue.append(face[i,2])
red=np.array(red).reshape((-1,1))
blue=np.array(blue).reshape((-1,1))
green=np.array(green).reshape((-1,1))
#print(red.shape,green.shape,blue.shape)
#print(face[2])
#print(red)
#print(face.shape[0])
result=np.stack((red,green,blue),axis=1)
result_df=pd.DataFrame(result,columns=["Channel 1","Channel 2","Channel 3"])
result_df.to_csv("Output.csv",index=False)
cv2.waitKey(0)
cv2.destroyAllWindows()
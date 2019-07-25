import cv2
import numpy as np
import os

# Algorithm***********************************************

def distance(x1,x2):
	return np.sqrt(np.sum((x1-x2)**2))
def KNN(X,Y,query_point,k=5):
	m=X.shape[0]
	val=[]
	for i in range(m):
		d=distance(X[i],query_point)
		val.append((d,Y[i]))
	val=sorted(val)
	val=val[:k]
	val=np.array(val)
	new_val=np.unique(val[:,1],return_counts=True)
	index=np.argmax(new_val[1])
	pred=new_val[0][index]
	return pred
#Data Preparation************************************************************

face_data=[]
labels=[]
names={}
dataset_path='./sample_data/'
class_id=0
skip=0
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(r'D:\\Coding Blocks Machine Learning\\Face_Recognition_project\\haarcascade_frontalface_alt.xml')
# Data Preparation **************************************************
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id]=fx[:-4]
		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)
		target=class_id*np.ones((data_item.shape[0],))
		labels.append(target)
		class_id+=1
face_dataset=np.concatenate(face_data,axis=0)
labels_dataset=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(labels_dataset.shape)

# Testing *****************************************************************
nose_cascade=cv2.CascadeClassifier(r'Nose18x15.xml')
eyes_cascade=cv2.CascadeClassifier(r'frontalEyes35x16.xml')
specs=cv2.imread('glasses.png',-1)
mustaches=cv2.imread('mustache.png',-1)
while True:
	ret,frame=cap.read()
	#if ret==False:
	#	continue
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	for (x,y,w,h) in faces:
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))
		face_section=np.array(face_section)
		out=int(KNN(face_dataset,labels_dataset,face_section.flatten()))
		pred_name=names[out]
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(255,86,78),2)
		#cv2.putText(frame,pred_name,(x,y-offset),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		noses=nose_cascade.detectMultiScale(frame,1.3,5)
		eyes=eyes_cascade.detectMultiScale(frame,1.3,5)
		frame=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
		mustaches=cv2.cvtColor(mustaches,cv2.COLOR_BGR2BGRA)
		specs=cv2.cvtColor(specs,cv2.COLOR_BGR2BGRA)
		for (nx,ny,nw,nh) in noses:
			#cv2.rectangle(frame,(nx,ny),(nx+nw,ny+nh),(0,255,255),2)
			mustaches=cv2.resize(mustaches,(nh+nh//2,nw))
			mw,mh,mx=mustaches.shape
			for i in range(0,mw):
				for j in range(0,mh):
					if mustaches[i][j][3]!=0:
						frame[ny+i+nw//2][nx+j-nw//4]=mustaches[i][j]	
		for (ex,ey,ew,eh) in eyes:
			#cv2.rectangle(frame,(ex,ey),(ex+eh,ey+ew),(0,255,255),2)
			specs=cv2.resize(specs,(eh,ew+ew//4))
			sw,sh,sc=specs.shape
			for i in range(0,sw):
				for j in range(0,sh):
					if specs[i][j][3]!=0:
						frame[ey+i-ew//15][ex+j+ew//9]=specs[i][j]
	cv2.imshow("Frame",frame)
	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

 


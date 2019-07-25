import cv2
import numpy as np
import os


cap=cv2.VideoCapture(0)

eyes_cascade=cv2.CascadeClassifier(r'D:\Coding Blocks Machine Learning\snapchat\frontalEyes35x16.xml')
nose_cascade=cv2.CascadeClassifier(r'D:\Coding Blocks Machine Learning\snapchat\Nose18x15.xml')





def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])






mustache = cv2.imread("mustache.png",-1)
glasses = cv2.imread("glasses.png",-1)







while True:
	ret,frame=cap.read()
	if ret==False:
		continue

	

	eyes=eyes_cascade.detectMultiScale(frame,1.3,5)
	nose=nose_cascade.detectMultiScale(frame,1.3,5)
	
	
	for (ex,ey,ew,eh) in eyes:
		# cv2.rectangle(frame,(ex,ey),(ex+eh,ey+ew),(0,255,255),2)

		eye = cv2.resize(glasses,(eh,ew))
		#  this function just insert one picture on top of other
		#  we had 2 images 1. -> frame  2. -> glasses
		#  superimpose glasses on frame, at this location (ex and ey)
		#  bcoz these were the corrdinates where haarcascade found your eyes. 
		#  simple...
		#  there are many others ways as well.....
		#  lekin sab me karna yehi hai ,,, glasses ko frame ke uppar chipka do at ex,ey location
		#  overlay_img_alpha() function ye jaadu karke de deta hai bs :-p
		#  slicing kse?
		#  uske liye function samjhna padega, function ko kya need hai..
		overlay_image_alpha(frame,eye[:,:,0:3],(ex,ey) ,eye[:,:,3]/255.)
		
		break


	for (nx,ny,nw,nh) in nose:
		# cv2.rectangle(frame,(nx,ny),(nx+nw,ny+nh),(0,255,255),2)
		mus = cv2.resize(mustache,(nh,nw))

		print(mus[:,:,0:3].shape)
		print((mus[:,:,3]/255.).shape)
		overlay_image_alpha(frame,mus[:,:,0:3],(nx,ny+(nh//2)),mus[:,:,3]/255.)

		break


	

	cv2.imshow("Frame",frame)
	
	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()

import os
import cv2
#images=os.listdir("data")
#images.sort()
#print(images)
import numpy as np
imgset =[]
for i in range(0,96):
   images = os.listdir("class/"+str(i))
   img = []
   for j,imgname in zip(range(0,90),images):
       img.append(cv2.imread("class/"+str(i)+"/"+imgname))
       if ((j+1)%9==0 ):
           final = np.zeros((28*3,28*3,3))
           final[:28,:28,:]=img[0]
           final[:28,28:28*2,:]=img[1]
           final[:28,28*2:28*3,:]=img[2]
           final[28:28*2,:28,:]=img[3]
           final[28:28*2,28:28*2,:]=img[4]
           final[28:28*2,28*2:28*3,:]=img[5]
           final[28*2:28*3,:28,:]=img[6]
           final[28*2:28*3,28:28*2,:]=img[7]
           final[28*2:28*3,28*2:28*3,:]=img[8]
           imgset.append(final)
           img = []
       
for i,image in zip(range(960),imgset):
    cv2.imwrite("frames/"+str(i)+".jpg",image)
       
           
       
    



#img1 = cv2.imread(imageFile1)
#img2 = cv2.imread(imageFile2)
#
#h1, w1 = img1.shape[:2]
#h2, w2 = img2.shape[:2]
#
##create empty matrix
#vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
#
##combine 2 images
#vis[:h1, :w1,:3] = img1
#vis[:h2, w1:w1+w2,:3] = img2
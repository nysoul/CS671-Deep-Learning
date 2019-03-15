from PIL import Image, ImageDraw
import numpy as np
import math
import random
import os
dir=os.system("mkdir data")
for i in range(0,2):
        if(i==0):
            length=7
        else:
            length=15
        for j in range(0,2):
            if(j==0):
                breadth=1
            else:
                breadth=3
            for k in range(0,12):
                theta=k*15*(math.pi)/180
                for l in range(0,2):
                    if(l==0):
                        color='red'
                    else:
                        color='blue'
                
                    for m in range(0,1000):
                        im1=np.zeros((28,28,3),dtype='int')
                        im=Image.fromarray(im1, 'RGB')
                        line1=ImageDraw.Draw(im)
                        y2=length*math.sin(theta)
                        x2=length*math.cos(theta)
                        
                        xshift=random.randint(-3,3)
                        yshift=random.randint(-3,3)
                        
                        line1.line((14- (x2/2) +xshift,14 + (y2/2) + yshift,14 + (x2/2) + xshift,14 - (y2/2) + yshift), fill = color, width=breadth)
                        
                        iname=str(i)+"_"+str(j)+"_"+str(k)+"_"+str(l)+"_"+str(m)+".jpg"
                        c = i*1+j*2+k*4+l*48
                       
                        im.save("data/"+iname) 
#                    
#import cv2
#import os
#
#image_folder = '/'
#video_name = 'video.avi'
#
#images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#frame = cv2.imread(os.path.join(image_folder, images[0]))
#height, width, layers = frame.shape
#
#video = cv2.VideoWriter(video_name, 0, 2, (width,height))
#
#for image in images:
#    video.write(cv2.imread(os.path.join(image_folder, image)))
#
#cv2.destroyAllWindows()
#video.release()
#                
                
                
                
                
    
        

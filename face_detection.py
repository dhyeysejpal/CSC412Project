#From : http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
import numpy as np
import cv2
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import matplotlib.patches as patches


face_cascade = cv2.CascadeClassifier('C:\\Users\\dhyey_000\\Downloads\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')

def get_detected_emotions(img):
    
    detections = []
    s = (0,1024)   #each image will be a 1024 length vector
    Y = np.empty(s)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    label = 'Neutral'

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  #detect the faces in the image using the cascade detector
    
    figure = plt.figure()
    axis = figure.add_subplot(111, aspect='equal')
    plt.axis('off')
    plt.imshow(img)

    for (x,y,w,h) in faces:
        f = gray[(y - 10):(y + h + 10), (x - 10):(x + w + 10)] #extract the face patch from the image
        im = get_processed_image(f)
        #TODO: Classify image, get the label
        axis.add_patch(patches.Rectangle((x-10,y-10),w+20,h+20, fill=None, ec=np.asarray([255., 0., 0.]) / 255., lw=3))
        plt.text(x, y, label, fontsize = 13, color = 'white', weight = 'heavy', backgroundcolor = 'red') #text for emotion
    
    plt.show()

def get_processed_image(img):
    f = np.zeros((img.shape[0],img.shape[1]))   # 2D array of shape same as the imput array
    f = imresize(img,(32,32)) #resize it to 32x32
    return f.reshape(1,1024) #flatten it as required for the input


if __name__ == '__main__':
    img = imread('C:\\Users\\dhyey_000\\Desktop\\CSC412\\CSC412Project\\0273.jpg')
    get_detected_emotions(img)
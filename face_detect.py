import cv2 as cv 

#You can use any photo you want

img = cv.imread('OpenCV\Photos\Fileame.jpg')
cv.imshow('aynen',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

haarCascade = cv.CascadeClassifier('OpenCV\Face Detection with Haar Cascades\haar_face.xml')



facesRect = haarCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
print(f'Number of faces = {len(facesRect)}')
for (x,y,w,h) in facesRect:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=2)



cv.imshow('FaceDetect',img)
cv.waitKey(0)

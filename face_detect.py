import cv2 as cv 

img = cv.imread('OpenCV\Photos\Aynen.jpg')
cv.imshow('aynen',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

haarCascade = cv.CascadeClassifier('OpenCV\Face Detection with Haar Cascades\haar_face.xml')



facesRect = haarCascade.detectMultiScale(gray,scaleFactor=1.1111111,minNeighbors=1)
print(f'Number of faces = {len(facesRect)}')
for (x,y,w,h) in facesRect:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=2)



cv.imshow('aynenFaces',img)
cv.waitKey(0)
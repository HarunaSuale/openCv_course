import cv2 
import numpy as np

# img = cv2.imread("./photos/ruwaida.jpg")

#convert an img from BGR to gray color space
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("normal img", img)
# cv2.imshow("Gray", gray)

# #convert img from BGR to RGB color space
# RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow('', RGB)

#Blur img with average blur
# ruwaida_blur = cv2.blur(img, (7,7))
# Gblur = cv2.GaussianBlur(img, (7,7), 1)
# cv2.imshow("normal Img", img)
# cv2.imshow('blur_image', ruwaida_blur)
# cv2.imshow('Gblur', Gblur)


#split BGR img to r, g, and b
# cv2.imshow('Haruna', img)
# r, g, b, = cv2.split(img)
# cv2.imshow("red", r)
# cv2.imshow('green', g)
# cv2.imshow('blue', b)


#create a blank img
# blank = np.zeros((500, 500), dtype='uint8')
# cv2.imshow('blank img', blank)

# circle = cv2.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 100, 255, -1)
# cv2.imshow("circle", circle)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# cv2.waitKey(0)    


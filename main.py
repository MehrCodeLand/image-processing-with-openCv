import cv2 as cv 
import numpy as np



# read image 
image = cv.imread('x.jpeg')

triangle = cv.imread('triangel.jpeg')
triangle = cv.cvtColor(triangle , cv.COLOR_BGR2GRAY)

#change color format
half_image = cv.resize(image, (0 ,0) , fx = 0.3 , fy = 0.3)
half_image = cv.cvtColor(half_image, cv.COLOR_BGR2GRAY)



# morfology
kernel = np.ones((5,5) , np.uint8)
erod = cv.erode(half_image, kernel, iterations=11)
dilation = cv.dilate(half_image, kernel , iterations=1)
openning = cv.morphologyEx(half_image, cv.MORPH_OPEN , kernel)


# threshold
ret , threshold1 = cv.threshold(half_image , 120 , 255 , cv.THRESH_BINARY)
ret , threshold2 = cv.threshold(half_image, 120 , 255 , cv.THRESH_TOZERO)
ret, triangle = cv.threshold(triangle, 120 , 255 , cv.THRESH_BINARY_INV)

# filtring
kernel_sharp = np.array([
    [0 , -1, 0],
    [-1, 5, -1],
    [0 , -1, 0]
])
sharp = cv.filter2D(half_image , -1 , kernel_sharp)

# edge
edge = cv.Canny(half_image , 100 , 350 )


# contour (area , perimeters) and how can trace shap
contour , _ = cv.findContours(triangle , cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE )
for cnt in contour : 
    area = cv.contourArea(cnt)
    perimeters = cv.arcLength(cnt , True)
    print("area " , area)
    print("perimeters" , perimeters)

for cnt in contour : 
    (x,y) , radius = cv.minEnclosingCircle(cnt)
    center = (int(x) , int(y))
    radius = int(radius)
    cv.circle(triangle , center , radius , (36 , 255 , 12 ) , 5)


# SIFT algorythm and fetures matching
sift = cv.SIFT_create()
kp ,  des = sift.detectAndCompute(half_image , None )
sift_image = cv.drawKeypoints(half_image , kp , None)

batman1 = cv.imread('batman1.jpeg')
batman2 = cv.imread('batman2.jpeg')

batman1 = cv.resize(batman1, (0 ,0) , fx = 0.3 , fy = 0.3)
batman2 = cv.resize(batman2, (0 ,0) , fx = 0.3 , fy = 0.3)

kpBat1 , desBat1 = sift.detectAndCompute(batman1 , None)
kpBat2 , desBat2 = sift.detectAndCompute(batman2 , None)

bf = cv.BFMatcher(cv.NORM_L1 , crossCheck=True)
matches = bf.match(desBat1 , desBat2)
matches = sorted(matches , key=lambda x:x.distance)

draw_image = cv.drawMatches(batman1 , kpBat1 , batman2 , kpBat2 , matches , None , flags=2)

# image segmentation 

image_resize = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
r, thresh = cv.threshold(half_image , np.mean(half_image) , 255 , cv.THRESH_BINARY_INV)
seg_con, h = cv.findContours(thresh, cv.RETR_LIST , cv.CHAIN_APPROX_SIMPLE)
seg_snt = sorted(seg_con , key=cv.contourArea)

mask = np.zeros((half_image.shape[0] , half_image.shape[1]), dtype='uint8')
mas_Final = cv.drawContours(mask , [seg_con[-1]] , 0 , (255,255,255) , -1 )


final_image = cv.bitwise_and(half_image , half_image , mask=mas_Final)

# k-means 
import matplotlib.pyplot as plt

img2 = cv.imread('batman1.jpeg')
img2 = cv.cvtColor(img2 , cv.COLOR_BGR2RGB)


pixel = img2.reshape((-1,3))
pixel = np.float32(pixel)

ceraiteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER , 10 , 1.0 )
k = 3 
retval , labels , centers = cv.kmeans(pixel, k , None, ceraiteria , 10 , cv.KMEANS_RANDOM_CENTERS )

centers = np.uint8(centers)
segment_data = centers[labels.flatten()]
segment_image = segment_data.reshape((img2.shape))
plt.imshow(segment_image)
plt.show()
# show image


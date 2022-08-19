import cv2
print( cv2.__version__ )
import numpy as np
import matplotlib.pyplot as plt
# load images
image1=cv2.imread("D:\\image\\w11.png")
image2=cv2.imread("D:\\image\\w1.png")
def wb(channel, perc = 0.05):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel
print(image2.shape)
image2  = np.dstack([wb(channel, 0.05) for channel in cv2.split(image2)] )
image1=cv2.resize(image1,(400,600))
image2=cv2.resize(image2,(400,600))

cv2.imshow("image1",image1)
cv2.imshow("image2",image2)
#Detect and compute interest points and their descriptors
def align(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:

            good.append(m)
    if len(good) > 2 :
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1,2)
                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.5)
                img_w = cv2.warpPerspective(img1,matrix,(img1.shape[1],img1.shape[0]))#move  old picture to match the new one
                img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None)
                cv2.imshow("matches",img3)
    return img_w
#mask image
def mask_img (image,lower,upper):
    image =cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image,lower,upper)
    #image = cv2.GaussianBlur(image,(5,5),0)
    kernel = np.ones((5, 5),np.uint8)
    #image = cv2.morphologyEX = (image, cv2.MORPH_OPEN, kernel)
    # Define the structuring element
   # kernel =
    # Apply the opening operation
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image= cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # Apply the closing operation
    #closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("mask1",image1)
    return image

#Align image
img_al=align(image1,image2)
cv2.imshow("image_aligned",img_al)
#Rang of colours(pink,white)
lower_pink = np.array([137,59,96])
#lower_pink = np.array([0,21,44])
upper_pink = np.array([179,255,255])
#upper_pink = np.array([255,132,148])
img_pink_image1 = mask_img(img_al,lower_pink,upper_pink)
img_pink_image2 = mask_img(image2,lower_pink,upper_pink)
#lower_white =  np.array([12,23,0])
lower_white = np.array([0,0,200])
#upper_white =  np.array([128,255,255])
upper_white = np.array([150,255,255])

img_white_image1= mask_img(img_al,lower_white,upper_white)
img_white_image2 = mask_img(image2,lower_white,upper_white)

cv2.imshow("only pink",img_white_image1)
cv2.imshow("only white",img_white_image2)

#Addition  image
image1_add_pink_white=cv2.add(img_pink_image1,img_white_image1)
cv2.imshow("ADD Image1",image1_add_pink_white)
image2_add_pink_white=cv2.add(img_pink_image2,img_white_image2)
cv2.imshow("ADD Image2",image2_add_pink_white)


cv2.imshow("only pink image2",img_pink_image2)
cv2.imshow("only white image1",img_pink_image1)
# subtract
image_cut_colours=cv2.subtract(img_white_image1,img_white_image2)
kernel = np.ones((10, 10), np.uint8)
image_cut_colours = cv2.morphologyEx(image_cut_colours, cv2.MORPH_OPEN, kernel)
image_cut_colours = cv2.morphologyEx(image_cut_colours, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Only colours",image_cut_colours)
image_cut_coral=cv2.bitwise_xor(image1_add_pink_white,image2_add_pink_white)
kernel = np.ones((10, 10), np.uint8)
image_cut_coral = cv2.morphologyEx(image_cut_coral, cv2.MORPH_OPEN, kernel)
image_cut_coral = cv2.morphologyEx(image_cut_coral, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Cut for coral",image_cut_coral)
image_cut_coral_two=cv2.bitwise_xor(image2_add_pink_white,image1_add_pink_white)
kernel = np.ones((10, 10), np.uint8)
image_cut_coral_two = cv2.morphologyEx(image_cut_coral_two, cv2.MORPH_OPEN, kernel)
image_cut_coral_two = cv2.morphologyEx(image_cut_coral_two, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Cut for coral two",image_cut_coral_two)
retval_2, binary_2= cv2.threshold(image_cut_coral, 225, 225, cv2.THRESH_BINARY)
contours_2= cv2.findContours(binary_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contours_image_2 = np.copy(image2)
for cnt in contours_2:
    area = cv2.contourArea(cnt)
    print(area)
    areaMin = cv2.getTrackbarPos("Area", "Parameters")
    #if area > 50:
        #cv2.drawContours(contours_image, cnt, -1, (255, 0, 255), 7)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    print(len(approx))
    x, y, w, h = cv2.boundingRect(approx)
    print(x)
    cv2.rectangle(contours_image_2, (x, y), (x + w, y + h), (0, 230, 0), 3)


masked_1 = image_cut_coral[x:x+w,y:y+h]
masked_2 = image2_add_pink_white[x:x+w,y:y+h]
cv2.imshow("masked 2",image2_add_pink_white)
coral=masked_1&masked_2



retval_2, binary_2= cv2.threshold(coral, 225, 225, cv2.THRESH_BINARY)
contours_2= cv2.findContours(binary_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contours_image_2 = np.copy(image2)
if len(contours_2) == 0 :
    color = (0,255,0)
else :
    color = (0,255,255)

print(len(contours_2))
for cnt in contours_2:
    area = cv2.contourArea(cnt)
    print(area)
    areaMin = cv2.getTrackbarPos("Area", "Parameters")
    #if area > 50:
        #cv2.drawContours(contours_image, cnt, -1, (255, 0, 255), 7)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
    print(len(approx))
    x, y, w, h = cv2.boundingRect(approx)
    print(x)
    cv2.rectangle(image2, (x, y), (x + w, y + h), color, 3)


cv2.imshow("masked 1",masked_1)
cv2.imshow("masked 2",masked_2)
cv2.imshow("coral",coral)
#coral_death=masked_1&image2_add_pink_white
#coral_growth=masked_2&image2_add_pink_white
#cv2.imshow("growth",coral_death)
#cv2.imshow("death",coral_growth)

# image_after[mask[mask!=0]] == mask[mask[mask!=0]]
'''
if  image2[mask_img(image_cut_coral_two,lower_pink,upper_pink)]!=0:
    cv2.imshow("Cut for coral two", image_cut_coral_two)
else:
    cv2.imshow("Cut for coral two", image_cut_coral_two)
'''
#image2[mask_img(image_cut_coral_two,lower_pink,upper_pink)[mask_img(image_cut_coral_two,lower_pink,upper_pink)!=0]]==mask_img(image_cut_coral_two,lower_pink,upper_pink)[]
#if image_cut_coral_two[mask_img(image_cut_coral_two,lower_pink,upper_pink)]!=0:
           #cv2.imshow("Cut for coral two", image_cut_coral_two)
#else:

    #cv2.imshow("Cut for coral two", image_cut_coral_two)


image2_cut_white_pink=cv2.bitwise_xor(img_white_image1,img_white_image2)
kernel = np.ones((10, 10), np.uint8)
image2_cut_white_pink = cv2.morphologyEx(image2_cut_white_pink, cv2.MORPH_OPEN, kernel)
image2_cut_white_pink = cv2.morphologyEx(image2_cut_white_pink, cv2.MORPH_CLOSE, kernel)

cv2.imshow("CUT coral two",image2_cut_white_pink)
#find contours colours red
#retval_3, binary_3 = cv2.threshold(image2_cut_white_pink, 225, 225, cv2.THRESH_BINARY)
contours_3= cv2.findContours(image2_cut_white_pink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contours_image = np.copy(image2)
for cnt in contours_3:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image2 ,(x, y), (x + w, y + h), (0, 0, 255),3)
#find contours colours yellow
#retval_3, binary_3 = cv2.threshold(image2_cut_white_pink, 225, 225, cv2.THRESH_BINARY)
contours_4= cv2.findContours(image_cut_coral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contours_image = np.copy(image2)
for cnt in contours_4:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image2 ,(x, y), (x + w, y + h), color,3)
#find contours colours blue
retval, binary = cv2.threshold(image_cut_colours, 225, 225, cv2.THRESH_BINARY)
contours= cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contours_image = np.copy(image2)
for cnt in contours:
    area = cv2.contourArea(cnt)
    print(area)
    areaMin = cv2.getTrackbarPos("Area", "Parameters")
    #if area > 50:
        #cv2.drawContours(contours_image, cnt, -1, (255, 0, 255), 7)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    print(len(approx))
    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(contours_image ,(x, y), (x + w, y + h), (255, 0, 0),3)

#find contours coral green
'''

retval_2, binary_2= cv2.threshold(coral, 225, 225, cv2.THRESH_BINARY)
contours_2= cv2.findContours(binary_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
contours_image_2 = np.copy(image2)
for cnt in contours_2:
    area_2 = cv2.contourArea(cnt)
    print(area_2)
    areaMin_2 = cv2.getTrackbarPos("Area", "Parameters")
    #if area > 50:
        #cv2
    # .drawContours(contours_image, cnt, -1, (255, 0, 255), 7)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    print(len(approx))
    x, y, w, h = cv2.boundingRect(approx)

    cv2.rectangle(contours_image ,(x, y), (x + w, y + h), (0, 230, 0),3)
'''
#c = contours[0]
#area = cv2.contourArea(c)
#print(area)
#if 150000 < area > 200000:
    #contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)
#print(type(contours))
cv2.imshow("contours",contours_image)


#cap = cv2.VideoCapture('path of video file')


cv2.waitKey(0)
cv2.destroyAllWindows()
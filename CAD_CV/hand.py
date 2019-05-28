import cv2
import numpy as np
import copy
from pynput.mouse import Button, Controller
import math

ionitArea = 0
mouse = Controller()
def movemouse():
    mouse.position = (1000,1000)
    mouse.press(Button.left)
    mouse.release(Button.left)
    print("HIIII")

cam = cv2.VideoCapture(0)
cv2.namedWindow("testHand")
def createCont(mCont):
    newCont = mCont
    #print(gesture(maxCont))
    #print("NEW")
    hull = cv2.convexHull(mCont,returnPoints=True)
    cv2.drawContours(img,[hull],0,(0,0,255),3)
    inc = 0
    crossx = 0
    crossy = 0
    hull = cv2.convexHull(mCont,returnPoints=False)
    #cv2.drawContours(img,[hull],0,(0,0,255),3)
    if(len(hull) > 3):
        diff = cv2.convexityDefects(mCont,hull)
        if diff is not None:
            f1,f2,f3,d = diff[diff.shape[0]-1][0]
            testPoint = tuple(mCont[f3][0])
            crossy = mCont[f3][0][1]
            crossx = mCont[f3][0][0]
            np.empty_like(newCont)
            cv2.circle(img,testPoint,8,[30,30,255],-1) #red
            for i in range(maxCont.shape[0]):
                if(mCont[i][0][1]<=testPoint[1]):
                    newCont[inc]=(tuple(mCont[i][0]))
                    inc+=1
            '''
            while(crossx>maxCont[f3][0][0]-50 and inc <maxCont.shape[0]):
                tup = (crossx,crossy)
                newCont[inc]=(tuple(tup))
                crossx-=1
                inc+=1
            '''
    return newCont
def getCircleArea(mCont):
    (x,y),radius = cv2.minEnclosingCircle(mCont)
    center = (int(x),int(y))
    radius = int(radius)
    return radius*radius*3.14
def gesture(mCont):
    fin = 1
    hull = cv2.convexHull(mCont,returnPoints=False)
    if(len(hull) > 3):
        diff = cv2.convexityDefects(mCont,hull)
        if diff is not None:
            for i in range(diff.shape[0]):
                f1,f2,f3,d = diff[i][0]
                p1 = tuple(maxCont[f1][0])
                p2 = tuple(maxCont[f2][0])
                p3 = tuple(maxCont[f3][0])
                #cv2.circle(img,p1,8,[255,84,0],-1) #dark blue
                #cv2.circle(img,p2,8,[255,255,255],-1) #white
                #cv2.circle(img,p3,8,[255,255,0],-1) #light blue
                a = math.sqrt(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
                b = math.sqrt(((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2))
                c = math.sqrt(((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2))
                angle = math.acos(((b*b+c*c-a*a)/(2*b*c)))
                if(i == 0):
                    if(diff.shape[0]>=3):
                        f1,f2,f3,d = diff[1][0]
                        testPoint = tuple(maxCont[f3][0])
                        newy = testPoint[1]
                        newx = testPoint[0]
                    #print("p3",newx,newy)
                #print("p1",p1[0],p1[1])
                #print("p2",p2[0],p2[1])
                #print("p3",p3[0],p3[1])
                #print("NEXT")
                #print(angle) 
                if(angle<math.pi/2):
                    fin+=1
                #do some mathy things
    nCont = createCont(mCont)
    '''
    if(fin == 1 and abs(initArea-cv2.contourArea(nCont))>=100):
        fin-=1
    '''
    return fin

def subimg(frame):
  fmask = first_frame.apply(frame)
  kernel = np.ones((3,3),np.uint8)
  fmask = cv2.erode(fmask,kernel,iterations=1)
  return cv2.bitwise_and(frame, frame, mask=fmask)
thresh_Val = 20
cap = False
first_frame = cv2.cvtColor(cam.read()[1],cv2.COLOR_RGB2GRAY)


cur = first_frame
while cam.isOpened():
  cur = cam.read()[1]
  cur = cv2.bilateralFilter(cur,5,50,100)
  #cv2.rectangle(cur, (int(.3 * cur.shape[1]), 5),(cur.shape[1], int(.8 * cur.shape[0])), (0, 0, 0), 5)
  cv2.imshow("orig",cur)
  k = cv2.waitKey(10)
  if k == 27:
    break
  elif k == ord('s'):
    first_frame = cv2.BackgroundSubtractorMOG2(0,50)
    cap = True
  if cap:
    img = subimg(cur)
    #img = img[0:int(.8*cur.shape[0]), int(.3*cur.shape[1]):cur.shape[1]]
    ret,thresh = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),thresh_Val,255,cv2.THRESH_BINARY)
    cv2.imshow("threshed", thresh)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    maxArea = 0
    index = -1
    length = len(contours)
    if(length>0):
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                index = i
   

    #cv2.drawContours(img,contours,-1,(0,255,0),3)
    if index == -1:
        continue
    if k == ord('c'):
        initArea = cv2.contourArea(maxCont)
    maxCont = contours[index]
    fin = gesture(maxCont)
    #newCont = maxCont
    newCont = maxCont
    #print(cv2.contourArea(contours[index]))
            
    #print("Change",newx-oldx,newy-oldy)
    #print((newx-40)*2.22,(newy-40)*1.9)
    #print(mouse.position)
    
    (x,y),radius = cv2.minEnclosingCircle(newCont)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(img,center,radius,[255,255,0],3) #light blue
    cv2.circle(img,center,8,[255,255,0],-1) #light blue
    cv2.drawContours(img,newCont,-1,(100,200,200),3) #yellow
    print(fin)
    '''
    cv2.drawContours(img,maxCont,0,(0,0,0),-1)
    #print(gesture(maxCont))
    '''
    #if(fin == 1):
    #    mouse.position=(int((center[0]-40)*2.22),int((center[1]-40)*1.9))
    cv2.imshow("mask",img)
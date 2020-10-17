#!/usr/bin/env python3
...

import math

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def hlsSplitting(image):
    converted = convert_hls(image)
    h,l,s = cv2.split(converted)
    return s,l

def select_black_yellow(image):
    
    masked = cv2.inRange(image,20,100)
    
    return masked

def apply_smoothing(image, kernel_size=3):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)

def edges (l_rMasked,x,y) :
    sobel = cv2.Sobel(l_rMasked , cv2.CV_64F, x,y)
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    return scaled_sobel

def angle(rect):
    """
    Produce a more useful angle from a rotated rect. This format only exists to make more sense to the programmer or
    a user watching the output. The algorithm for transforming the angle is to add 180 if the width < height or
    otherwise add 90 to the raw OpenCV angle.
    :param rect: rectangle to get angle from
    :return: the formatted angle
    """
    if rect[1][0] < rect[1][1]:
        return rect[2] + 180
    else:
        return rect[2] + 90

def distance(a, b):
    """
    Calculate the distance between points a & b
    :return: distance as given by the distance formula: sqrt[(a.x - b.x)^2 + (a.y - b.y)^2]
    """
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def hull_score(hull):
    """
    Give a score to a convex hull based on how likely it is to be a qualification gate element.
    :param hull: convex hull to test
    :return: Score based on the ratio of side lengths and a minimum area
    """
    rect = cv2.minAreaRect(hull)
    shorter_side = min(rect[1])
    longer_side = max(rect[1])

    # the orange tape marker is 3 inches by 4 feet so the ratio of long : short side = 16
    ratio_score = 1 / (abs((longer_side / shorter_side) - 16) + 0.001)  # add 0.001 to prevent NaN

    score = ratio_score + cv2.contourArea(hull)

    # cut off minimum area at 500 px^2
    if cv2.contourArea(hull) < 500:
        return 0

    return score



def hullx_score(hull):

  score = 0

  rect = cv2.minAreaRect(hull) 
  short = min(rect[1])
  longe = max(rect[1])

  score += cv2.contourArea(hull)  # score based on area

  if (longe >= 12 * short) and (longe <= 19 * short) :
    score += 500    # score based on aspect ratio

  if (angle(rect) in range(180-15 , 180 + 15)) or (angle(rect) in range(0-15 , 0 + 15)) :
    score += 200      # score based on angle

  return score


def hully_score(hull,left,right):

  score = 0

  rect = cv2.minAreaRect(hull) 
  short = min(rect[1])
  longe = max(rect[1])
  cx = int (rect[0][0])
  cy = int (rect[0][1])
  theta = angle(rect)
  

  score += cv2.contourArea(hull)

  if cx in range(int(left[0][0]),int(right[0][0])):
    score += 5000
  if (cy < int(left[0][1])) and (cy < int(right[0][1])): 
    score += 5000
  if abs(angle(right) - theta ) in range (90-10 , 90+10):
    score += 5000
  if abs(angle(left) - theta ) in range (90-10 , 90+10):
    score += 5000 

  return score

def hulls_score(hull,left,right,horizontal): 

  score = 0

  rect = cv2.minAreaRect(hull)
  short = min(rect[1])
  longe = max(rect[1])
  cx = int (rect[0][0])
  cy = int (rect[0][1])
  theta = angle(rect)

  if cx in range(int(left[0][0]),int(right[0][0])):
    score += 5000
  if cy > horizontal[0][1]:
    score += 5000
  if cy in range (int(horizontal[0][1]),int(right[0][1])):
    score += 5000
  if abs( angle(right) - theta ) in range (0-5 , 5 ):
    score += 5000
  if distance(right[0],rect[0]) < distance(right[0],left[0]):
    score += 5000
  if (longe >= 4*short) and (longe <= 9*short):
    score += 5000

  return score    

def contouring (imagex,imagey):
    contoursx, hierarchyx = cv2.findContours(imagex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contoursy, hierarchyx = cv2.findContours(imagey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return contoursx, contoursy
   

def convex_hulls(contours):
    """
    Convenience method to get a list of convex hulls from list of contours
    :param contours: contours that should be turned into convex hulls
    :return: a list of convex hulls that match each contour
    """

    hulls = []
    for contour in contours:
        hulls.append(cv2.convexHull(contour))

    return hulls

def gateParts(hullsx,hullsy):
  hullsx.sort(key = hullx_score)
  left = cv2.minAreaRect(hullsx[-1])
  right = cv2.minAreaRect(hullsx[-2])

  if right[0][0] < left[0][0]:
    left, right = right, left

  scoresy = []
  for i in hullsy:
    scoresy.append(hully_score(i,left,right))

  horizontal = cv2.minAreaRect(hullsy[np.argmax(scoresy)])

  scoress = []
  for j in hullsx[0:-2]:
    scoress.append(hulls_score(j,left,right,horizontal))

  small = cv2.minAreaRect(hullsx[np.argmax(scoress)])  
  # print(small,right)



  return right,left,small,horizontal

def bounsBox(img,right,left,small,horizontal):
  cx = int(np.median([small[0][0],left[0][0]]))
  cy = int(np.median([right[0][1],left[0][1]]))
  ty = min(horizontal[1])
  tx = min(left[1])
  X = small[0][0]- left[0][0]
  Y = max(left[1])
  start = (int(cx-(X/2)-tx) , int(cy-(Y/2)-ty))
  end   = (int(cx+(X/2)+tx) , int(cy+(Y/2)+ty))
  bigBox(img,right,left,small,horizontal)

  cv2.rectangle(img,start,end,(255,0,0),3)
  #print(small[0][0],small[0][1])
  return  img

def bigBox(img,right,left,small,horizontal):
  cx = int(np.median([right[0][0],left[0][0]]))
  cy = int(np.median([right[0][1],left[0][1]]))
  ty = min(horizontal[1])
  tx = min(right[1])
  X = right[0][0]- left[0][0]
  Y = max(right[1])
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  start = (int(cx-(X/2)-tx) , int(cy-(Y/2)-2*ty))
  end   = (int(cx+(X/2)+tx) , int(cy+(Y/2)+ty))
  cv2.rectangle(img,start,end,(255,255,0),3)
  return img

def imageProcessing(image):


  #cv2.resize(image, (600,800)) 
  cv2.line(image, (int(image.shape[1]/3), 0),(int(image.shape[1]/3), image.shape[0]), (0, 255,255), 1, 1)
  cv2.line(image, (int(image.shape[1]/1.5), 0),(int(image.shape[1]/1.5), image.shape[0]), (0, 255,255), 1, 1)
  cv2.line(image, (0,int(image.shape[0]/3)),(image.shape[1],int( image.shape[0]/3)), (0, 255,255), 1, 1)
  cv2.line(image, (0,int(image.shape[0]/1.5)),(image.shape[1],int( image.shape[0]/1.5)), (0, 255,255), 1, 1)
  s,l=hlsSplitting(image)
  sMask=select_black_yellow(s)
  lMask=select_black_yellow(l)
  r,c = lMask.shape[0],lMask.shape[1]
  lMask[ r//2 : -1 , : ] = 255
  verticalEdges=edges(sMask,1,0)
  horizentalEdges=edges(lMask,0,1)
  contourVert,contourHor=contouring(verticalEdges,horizentalEdges)
  hullsx = convex_hulls(contourVert)
  hullsy = convex_hulls(contourHor)
  right,left,small,horizontal=gateParts(hullsx,hullsy)
  bounsBox(image,right,left,small,horizontal)
 # motionStsps((int)left[0][0],(int)left[0][1])
   #plt.imshow(image)
  return left[0][0], left[0][1]

def gatenumber(image):
   y,x =imageProcessing(image)
   a=5
   return a;

import rclpy
from rclpy.node import Node
import sys
from std_msgs.msg import String
import numpy as np
import PIL 
import cv2



image = cv2.imread(sys.argv[1])
class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        

        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.i=gatenumber(image)
        msg1 = String()
        msg1.data = '%d' % self.i
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.publisher_.publish(msg1)
        
        self.get_logger().info('Publishing: "%s"' % msg1.data)
 
        self.subscription = self.create_subscription(String,'topic1',self.listener_callback,10)
      
        self.subscription  # prevent unused variable warning
	
    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
       

    def timer_callback(self):
        msg = String()
        msg.data = ' %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



#from moviepy.editor import *


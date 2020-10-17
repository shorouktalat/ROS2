import rclpy
from rclpy.node import Node
import sys
from std_msgs.msg import String
import numpy as np
import PIL 
import cv2
import math
from PIL import Image
from PIL import ImageEnhance
from pylab import rcParams
from matplotlib import pylab
import matplotlib as plt
from matplotlib import pyplot as plt

#image = cv2.imread(sys.argv[1])
#image = cv2.imread("/home/ahmed/noise1.png")
#image = Image.open(sys.argv[1])

def getGateGridNumber(image):
    imageEnhanced = PIL.ImageEnhance.Brightness(image).enhance(0.9)
    imageEnhanced = PIL.ImageEnhance.Contrast(imageEnhanced).enhance(2)
    imageEnhanced = PIL.ImageEnhance.Sharpness(imageEnhanced).enhance(3.2)

    imageCV = cv2.cvtColor(np.array(imageEnhanced), cv2.COLOR_RGB2BGR)

    b, g, r = cv2.split(imageCV)
    b = cv2.medianBlur(b,5)

    kernel = np.ones((5,5), np.uint8)
    b = cv2.erode(b, kernel, iterations=5)
    b = cv2.dilate(b, kernel, iterations=5)

    rcParams['figure.figsize'] = 10, 12
    edges = cv2.Canny(b, 
                    threshold1=100, ## try different values here
                    threshold2=100) ## try different values here
                    
    #plt.title('Edge Detection')
    #plt.imshow(edges)
    #plt.show()

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    imageOutput = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    height, width = edges.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    contoursFiltered = []
    # computes the bounding box for the contour, and draws it on the frame,
    i = 0
    rectanglesWidths = []
    rectanglesHeights = []
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        color = (0, 255, 0) if (i == 0) else (0,0,255)
        if w > 100 and h > 50:
            h = 250
            cv2.rectangle(imageOutput, (x,y), (x+w,y+h), color, 2)
            contoursFiltered.append(contour)
            i += 1
            # Saving widths and heights to later on put text on the center of the rectangles
            width = x+(w/2)
            height = y+(h/2)
            rectanglesWidths.append(width)
            rectanglesHeights.append(height)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(imageOutput, (min_x +50, min_y +150), (max_x, max_y), (255, 0, 0), 2)

    if len(rectanglesWidths)>=2:
        for i in range(len(rectanglesWidths)):
            if (rectanglesWidths[0] * rectanglesHeights[0] > rectanglesWidths[1] * rectanglesHeights[1]):
                cv2.putText(imageOutput, text= 'Big Gate', org=(int(rectanglesWidths[0]-90),int(rectanglesHeights[0])),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
                    thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(imageOutput, text= 'Small Gate', org=(int(rectanglesWidths[1]-90),int(rectanglesHeights[1])),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
                    thickness=2, lineType=cv2.LINE_AA)
            else:
                cv2.putText(imageOutput, text= 'Big Gate', org=(int(rectanglesWidths[1]-90),int(rectanglesHeights[1])),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
                    thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(imageOutput, text= 'Small Gate', org=(int(rectanglesWidths[0]-90),int(rectanglesHeights[0])),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
                    thickness=2, lineType=cv2.LINE_AA)

    #plt.title("Gate Detection")
    #plt.imshow(cv2.cvtColor(imageOutput, cv2.COLOR_BGR2RGB))
    #plt.show()

    if len(rectanglesWidths)>=2:
        for i in range(len(rectanglesWidths)):
            if (rectanglesWidths[0] * rectanglesHeights[0] < rectanglesWidths[1] * rectanglesHeights[1]):
                centerX = rectanglesWidths[0]
                centerY = rectanglesHeights[0]
            else:
                centerX = rectanglesWidths[1]
                centerY = rectanglesHeights[1]

    #print(centerX, centerY)

    #Code to detect which grid (1 to 9)
    #Send the integer to the C++ node
    height, width = edges.shape
    #print(height, width)
    hThird = height / 3
    wThird = width / 3
    if (centerX >= 0 and centerY >= 0 and centerX <= wThird and centerY <=hThird):
        grid = 1
    elif (centerX >= wThird and centerY >= 0 and centerX <= wThird*2 and centerY <=hThird):
        grid = 2
    elif (centerX >= wThird*2 and centerY >= 0 and centerX <= wThird*3 and centerY <=hThird):
        grid = 3
    elif (centerX >= 0 and centerY >= hThird and centerX <= wThird and centerY <=hThird*2):
        grid = 4
    elif (centerX >= wThird and centerY >= hThird and centerX <= wThird*2 and centerY <=hThird*2):
        grid = 5
    elif (centerX >= wThird*2 and centerY >= hThird and centerX <= wThird*3 and centerY <=hThird*2):
        grid = 6
    elif (centerX >= 0 and centerY >= hThird*2 and centerX <= wThird*1 and centerY <=hThird*3):
        grid = 7
    elif (centerX >= wThird and centerY >= hThird*2 and centerX <= wThird*2 and centerY <=hThird*3):
        grid = 8
    elif (centerX >= wThird*2 and centerY >= hThird*2 and centerX <= wThird*3 and centerY <=hThird*3):
        grid = 9

    return grid

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        


        self.declare_parameter("img_src")
        image = Image.open(self.get_parameter("img_src").get_parameter_value().string_value)

       # image = cv2.imread(self.get_parameter("img_src").get_parameter_value().string_value)

        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.i=getGateGridNumber(image)
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


#### Main starts here
#image = Image.open("C:/Users/Lenovo/Desktop/Vortex AUV/testing images/noise1.png")
#imagePath = Image.open(sys.argv[1])
#print(getGateGridNumber(imagePath))



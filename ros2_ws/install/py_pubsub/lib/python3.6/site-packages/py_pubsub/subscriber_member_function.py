# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

msg1 = String()
class MinimalSubscriber(Node):

#def direction ()
    
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(String,'topic',self.listener_callback,10)
        
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(String, 'topic1', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
         
    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        msg1.data=msg.data;
    

    def timer_callback(self):
        msg = String()
        msg.data="dd"
        if(msg1.data==' 1'):
           msg.data="left + up"
        elif(msg1.data==' 2'):
           msg.data = 'up'
        elif(msg1.data==' 3'):
           msg.data = 'right + up'
        elif(msg1.data==' 4'):
           msg.data = 'left'

        elif(msg1.data==' 5'):
           msg.data = 'center'
        elif(msg1.data==' 6'):
            msg.data = 'right'
        elif(msg1.data==' 7'):
           msg.data = 'down+left'
        elif(msg1.data==' 8'):
           msg.data = 'up'
        else :
            msg.data = 'down+right'

        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        
def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

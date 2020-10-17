#include "my_cpp_py_pkg/cpp_header.hpp"
#include <string.h>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
using namespace std::chrono_literals;
using std::placeholders::_1;

//#include <sstream>
using namespace std;
 


class MyNode : public rclcpp::Node
{
  public:

    MyNode() : Node("my_node"),count_(0)
    {
  
	
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "topic", 10, std::bind(&MyNode ::topic_callback, this, _1));
	 publisher_ = this->create_publisher<std_msgs::msg::String>("topic1", 10);
      timer_ = this->create_wall_timer(
      5000ms, std::bind(&MyNode::timer_callback, this));
     
    }

private:

void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
	
      RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
      //auto message = std_msgs::msg::String();
	auto message = std::string();
	//auto message = std::string();   
    	if(!strcmp(msg->data,"1"))
  
 		message="Right + Down"; 
   
	 else if(!strcmp(msg->data,"2"))
	 	message= "Down";
    	else if(msg->data=="3")
 		message= "Left + Down"; 
    	else if(msg->data=="4")
		message="Right"; 
    	else if(msg->data=="5") 
		message="Already centered"; 
    	else if(msg->data=="6")
 		message= "Left"; 
    	else if(msg->data=="7")
 	message= "Right + Up"; 
    	else if(msg->data=="8")
 	message= "Up"; 
    	else 
	message="Left + Up"; 
   // return message
 	//message = Direction(msg);
	auto message1 = std_msgs::msg::String();
    message1.data = " world";
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message);
      publisher_->publish(message1);
       	
       	
    }
    void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "hh";
    //RCLCPP_INFO(this->get_logger(), "Publishing: '%s'",message.data.c_str());
    //publisher_->publish(message);
  }  
    
 

  

	
     rclcpp::TimerBase::SharedPtr timer_;
     rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
   size_t count_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
   //msgback =std_msgs::msg::String();
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MyNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}



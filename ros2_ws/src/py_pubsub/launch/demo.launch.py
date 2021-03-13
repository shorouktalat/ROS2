from launch import LaunchDescription
from launch_ros.actions import Node



def generate_launch_description():
    ld = LaunchDescription()
    
    talker_node = Node(
        package="py_pubsub",
        node_executable="talker",
        output='screen', 
        parameters=[{"img_src":"/home/shorouk/51.png"}]
        
    )
    listener_node = Node(
        package="py_pubsub",
        node_executable="listener",
        output='screen'
    )
    ld.add_action(talker_node)
    ld.add_action(listener_node)
    return ld

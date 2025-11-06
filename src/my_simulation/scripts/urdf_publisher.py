#!/usr/bin/env python3
# filepath: /home/linux/robo_workspace/src/my_simulation/scripts/urdf_publisher.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import os

class URDFPublisher(Node):
    def __init__(self, urdf_path, topic_name):
        super().__init__('urdf_publisher')
        
        # Read URDF content
        try:
            with open(urdf_path, 'r') as file:
                self.urdf_content = file.read()
                self.get_logger().info(f'Loaded URDF from: {urdf_path}')
                self.get_logger().info(f'URDF content length: {len(self.urdf_content)} characters')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            return
        
        # Create publisher with latching behavior
        self.publisher = self.create_publisher(
            String, 
            topic_name, 
            rclpy.qos.QoSProfile(
                depth=10,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            )
        )
        
        # Timer to publish continuously
        self.timer = self.create_timer(1.0, self.publish_urdf)
        self.get_logger().info(f'Publishing URDF on topic: {topic_name}')

    def publish_urdf(self):
        msg = String()
        msg.data = self.urdf_content
        self.publisher.publish(msg)
        self.get_logger().info('Published URDF description')

def main():
    rclpy.init()
    
    if len(sys.argv) != 3:
        print("Usage: urdf_publisher.py <urdf_path> <topic_name>")
        return
    
    node = URDFPublisher(sys.argv[1], sys.argv[2])
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
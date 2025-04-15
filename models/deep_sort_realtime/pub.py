#!/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist


def publisher():
	rospy.init_node('pub_twist_node')

	pub = rospy.Publisher('cmd_vel',
						  Twist,
						  queue_size=10)

	rate = rospy.Rate(1)

	msg = Twist()
	msg.linear.x = 3 # 0.01
	msg.linear.y = 0.0
	msg.linear.z = 0.0
	msg.angular.x = 0.0
	msg.angular.y = 0.0
	msg.angular.z = 0.0

	while not rospy.is_shutdown():
		pub.publish(msg)
		rate.sleep()


def main():
	try:
		publisher()
	except rospy.ROSInterruptException:
		pass


if __name__ == "__main__":
	main()

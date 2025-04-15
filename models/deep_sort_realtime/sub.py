#!/usr/bin/env python

import cv2
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

num = 0
bridge = CvBridge()


def callback(msg):
	global num

	try:
		img = bridge.imgmsg_to_cv2(msg, 'bgr8')

		cv2.imshow('Camera', img)
		cv2.waitKey(100)
		cv2.imwrite('./data/{}.jpg'.format(num), img)

		rospy.loginfo('Save an image: {}'.format(num))
		num += 1
	except CvBridgeError as e:
		print(e)


def main():
	rospy.init_node('sub_camera_node')
	rospy.Subscriber('/camera/color/image_raw', Image, callback)
	rospy.spin()


if __name__ == "__main__":
	main()
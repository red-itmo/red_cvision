#!/usr/bin/env python2
import argparse
import rospy

from servers.start import Start

CAMERA_TOPICS = ['/usb_cam/image_rect', '/usb_cam/image_raw']


def getArgs():
    length = 0
    topicNum = 0
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--length', required=True,
                    help='length from camera to surface in [mm]')
    ap.add_argument('-t', '--topic', required=False,
                    help='camera topic num from array from talker.py')

    args = vars(ap.parse_args())
    length = float(args['length'])
    if args['topic'] is not None:
        topicNum = int(args['topic'])
    return length, topicNum


def talker():
    """ the main launching function """
    length, topicNum = getArgs()
    rospy.init_node('cv_recognizer', anonymous=False)
    rospy.loginfo("Camera's topic is " + CAMERA_TOPICS[topicNum])
    aaa = Start(CAMERA_TOPICS[topicNum], length)
    rospy.loginfo('Spining...')
    rospy.spin()
    del(aaa)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException, e:
        print("talker.py: " + e.message)

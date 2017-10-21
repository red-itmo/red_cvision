import rospy
import time
from sensor_msgs.msg import Image

from red_msgs.srv import CameraTask
from libs.utils import *
from libs.pymeasuring import Measuring


class CameraTaskServer:

    def __init__(self, cameraTopic, length):
        self.name = 'CameraTaskServer' + str(self)
        self.isReady = False

        self.cameraTopic = cameraTopic
        self.length = length

        self.imageInfo = dict()
        self.imageInfo['shape'] = None
        self.imageInfo['ratio'] = None  # ration [mm] / [px]

        self.measuring = None

        self.list = []  # list of found objects

        self.subCamera = None
        self.cameraTaskServer = None

        self.pubView = rospy.Publisher('see_main', Image, queue_size=1)

    def __del__(self):
        del(self.measuring)
        del(self.subCamera)
        del(self.cameraTaskServer)
        del(self.pubView)

    def init(self):
        """ initialization of need parameters"""
        imageDim = getDimImage(self.length, 0, 0, 78)  # 54.5, 42.3, 66.17
        self.imageInfo['ratio'] = getRatio(self.imageInfo['shape'], imageDim)
        self.measuring = Measuring(self.imageInfo, self.length)
        rospy.loginfo('init of measuring object is complete.')

    def camMode1(self, msgImage):
        print('camera callback')
        if self.isReady:
            cvImage, self.imageInfo['shape'] = getCVImage(msgImage)
            if self.measuring is not None:
                self.list, cvImage, self.isReady = self.measuring.getListObjects(cvImage)
                # preview topic /see_main
                msg_image = getMsgImage(cvImage)
                self.pubView.publish(msg_image)
            else:
                if self.imageInfo['shape'] is not None:
                    self.init()
                else:
                    rospy.logerr("no video stream. check camera's topic!")

    def handler(self, request):
        rospy.loginfo('Start processing request')
        self.list = []
        self.isReady = True
        if request.mode == 1:
            self.length = request.distance
            self.subCamera = rospy.Subscriber(self.cameraTopic, Image, self.camMode1)

            # if dont see anything
            starttime = time.time()
            curtime = 0
            while self.isReady:
                curtime = time.time() - starttime
                if curtime > 5.:
		    break
                print(curtime)
                time.sleep(0.01)

            print(self.list)
        elif request.mode == 2:
            pass
        self.reset()
        rospy.loginfo('End processing request')
        return 0, self.list

    def reset(self):
        rospy.loginfo('!!!reset!!! ' + str(self))
        self.subCamera.unregister()
        self.isReady = False
        self.cameraTaskServer = None
        self.measuring = None


    def startServer(self):
        self.cameraTaskServer = rospy.Service('camera_task', CameraTask, self.handler)
        rospy.loginfo('CameraTaskServer start!')

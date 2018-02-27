import rospy

from red_msgs.srv import CameraStop


class CameraStopServer:

    def __init__(self, cameraTaskServer):
        self.name = 'CameraStopServer' + str(self)

        self.stopServer = None
        self.cameraTaskServer = cameraTaskServer

    def __del__(self):
        del(self.cameraTaskServer)

    def handle(self, request):
        if request.mode == 0:
            self.cameraTaskServer.reset()
        return []

    def startServer(self):
        self.stopServer = rospy.Service('camera_stop', CameraStop, self.handle)
        rospy.loginfo('CameraStopServer start!')


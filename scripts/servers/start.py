#!/usr/bin/env python
import rospy
import threading

from camera_task_server import CameraTaskServer
from camera_stop_server import CameraStopServer


class ServerThread(threading.Thread):

    def __init__(self, server):
        threading.Thread.__init__(self)
        self.server = server
        self.isStop = False

    def __del__(self):
        del(self.server)

    def stop(self):
        self.isStop = True

    def run(self):
        self.server.startServer()
        rospy.loginfo(self.server.name + ' run!')
        while self.is_alive:
            if self.isStop:
                break


class Start:
    """ creation threads for servers"""

    def __init__(self, cameraTopic, length):
        rospy.loginfo('Start init threads')

        self.cts = CameraTaskServer(cameraTopic, length)
        self.css = CameraStopServer(self.cts)

        # self.cameraTaskTread = ServerThread(cts)
        # self.cameraTaskTread.start()

        self.cameraStopTread = ServerThread(self.css)
        self.cameraStopTread.start()

        self.cts.startServer()

        rospy.loginfo('End init threads')

    def __del__(self):
        self.cameraStopTread.stop()
        del(self.cameraStopTread)
        del(self.cts)
        del(self.css)


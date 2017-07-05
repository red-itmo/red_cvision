#!/usr/bin/env python
import rospy
import threading

from camera_task_server import CameraTaskServer
from camera_stop_server import CameraStopServer


class ServerThread(threading.Thread):

    def __init__(self, server):
        threading.Thread.__init__(self)
        self.server = server

    def run(self):
        self.server.startServer()
        rospy.loginfo(self.server.name + ' run!')
        while self.is_alive:
            # TODO something
            pass


class Start:
    """ creation threads for servers"""

    def __init__(self, cameraTopic, length):
        rospy.loginfo('Start init threads')

        cts = CameraTaskServer(cameraTopic, length)
        css = CameraStopServer(cts)

        self.cameraTaskTread = ServerThread(cts)
        self.cameraTaskTread.start()

        self.cameraStopTread = ServerThread(css)
        self.cameraStopTread.start()
        rospy.loginfo('End init threads')

    def hardReset(self):
        # TODO killing threads
        pass

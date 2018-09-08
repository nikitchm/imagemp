class FrameGrabberAbstract(object):
    def __init__(self):
        pass

    def init_unpickable(self):
        # Start grabber components which are not pickable, s.a. VideoCapture() in opencv
        pass

    def capture(self, shared_frame):
        # Should update the shared frame (image and timestamp)
        pass

    def close(self):
        pass

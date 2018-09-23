from __future__ import print_function
import cv2
from .abstract import *
import time


class FrameGrabberCV2(FrameGrabberAbstract):
    def __init__(self, file_device_name=0, init_unpickable=True, **kwargs):
        super(FrameGrabberCV2, self).__init__()
        # file_device_name corresponds to the argument of the cv2.VideoCapture() method -- an integer or string
        self.file_device_name = file_device_name
        self.timestamp = -1             # last timestamp value
        self.init_timestamp_method()
        self.iframe = None
        self.cap = None
        self.is_opened = False
        if init_unpickable:
            self.init_unpickable()

    def init_unpickable(self):
        # self.cap = cv2.VideoCapture(self.file_device_name)
        self.cap = self.gen_capture_obj()
        self.is_opened = self.open_for_capture(self.file_device_name)

    def gen_capture_obj(self):
        # split the generation with opening of the file to allow
        # for more versatile inheritance scenarios
        return cv2.VideoCapture()

    def open_for_capture(self, file_device_name):
        # file_device_name -- either a string with the path to a video file or
        # a camera ID (0,1,2,.. -- internal for opencv)
        return self.cap.open(file_device_name)

    @staticmethod
    def get_opencv_prop(obj, prop):
        # This function is taken from movies3 file
        import cv2
        """ Get the propery, prop, of the opencv opject, obj.
        Works for both, Python 2 and 3+ (that's the main reason for introducing this interface).
         """

        # print('params: {}'.format(prop))
        if int(cv2.__version__.split('.')[0]) < 3:
            val = obj.get(getattr(cv2.cv, 'CV_' + prop))
        else:
            val = obj.get(getattr(cv2, prop))
        return val

    def frame_i(self):
        return int(self.get_opencv_prop(self.cap, 'CAP_PROP_POS_FRAMES'))   # returns -1 if a camera

    def frame_ms(self):
        return self.get_opencv_prop(self.cap, 'CAP_PROP_POS_MSEC')   # returns -1 if a camera

    def get_timestamp(self):
        return self.frame_ms()

    def init_timestamp_method(self):
        # Allows to initialize the get_timestamp method in an overridden custom class
        pass

    def capture(self, shared_frame=None):
        # If shared_frame is not None, capture and write the next frame
        # (im and timestamp) into the shared_frame, otherwise, return the
        # next frame and timestamp (this way, the captured frame can be written
        # directly into the shared memory, avoiding an intermediate memory copy)
        _success = False
        if self.cap.isOpened():
            if shared_frame is not None:
                try:
                    # ret, shared_frame.im[:] = self.cap.read()
                    # t = time.time()
                    _success, shared_frame.im = self.cap.read()
                    # if not _success:
                    #     raise NameError("Couldn't read the frame.")
                    # ret, shared_frame.im[:] = self.cap.read()
                    # ret = self.cap.read(shared_frame.im[:])
                    # self.n += 1
                    # self.dtmean = (self.dtmean * (self.n-1) + time.time()-t) / self.n
                    # print('dt={}'.format(self.dtmean))
                except Exception as e:
                    print('Error while trying to read the frame: {}'.format(e))
                if _success:
                    self.timestamp = self.get_timestamp()
                    shared_frame.timestamp = self.timestamp
                return _success
            else:
                # Return the next captured frame
                return self.cap.read(), self.get_timestamp()

    def close(self):
        try:
            self.cap.release()
        finally:
            self.cap = None


class FrameGrabberCV2File(FrameGrabberCV2):
    def __init__(self, vid_filename, init_unpickable=True, **kwargs):
        super(FrameGrabberCV2File, self).__init__(file_device_name=vid_filename, init_unpickable=init_unpickable, **kwargs)

    def frame_i(self):
        return int(self.get_opencv_prop(self.cap, 'CAP_PROP_POS_FRAMES'))

    def frame_ms(self):
        return self.get_opencv_prop(self.cap, 'CAP_PROP_POS_MSEC')

    def get_timestamp(self):
        return self.frame_ms()


class FrameGrabberCV2Camera(FrameGrabberCV2):
    def __init__(self, device_id, init_unpickable=True, **kwargs):
        super(FrameGrabberCV2Camera, self).__init__(file_device_name=device_id, init_unpickable=init_unpickable, **kwargs)

    def frame_i(self):
        return -1

    def frame_ms(self):
        return -1

    def get_timestamp(self):
        return time.time()

from __future__ import print_function
import cv2
from abstract import *


class FrameGrabberCV2File(FrameGrabberAbstract):
    def __init__(self, vid_filename, init_unpickable=True, **kwargs):
        super(FrameGrabberCV2File, self).__init__()
        self.filename = vid_filename
        self.timestamp = -1             # last timestamp value
        self.init_timestamp_method()
        self.iframe = None
        self.cap = None
        self.is_opened = False
        if init_unpickable:
            self.init_unpickable()

    def init_unpickable(self):
        # self.cap = cv2.VideoCapture(self.filename)
        self.cap = self.gen_capture_obj()
        self.is_opened = self.open_file_for_capture(self.filename)

    def gen_capture_obj(self):
        return cv2.VideoCapture()

    def open_file_for_capture(self, filename):
        # split the generation with opening of the file to allow
        # for more versatile inheritance scenarios
        return self.cap.open(filename=filename)

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
        return int(self.get_opencv_prop(self.cap, 'CAP_PROP_POS_FRAMES'))

    def frame_ms(self):
        return self.get_opencv_prop(self.cap, 'CAP_PROP_POS_MSEC')

    def get_timestamp(self):
        return self.frame_ms()

    def init_timestamp_method(self):
        # Allows to initialize the get_timestamp method in an overridden custom class
        pass

    def capture(self, shared_frame=None):
        # If shared_frame is not None, capture and write the next frame
        # (im and timestamp) into the shared_frame,
        # otherwise, return the next frame and the timestamp
        ret = False
        if self.cap.isOpened():
            if shared_frame is not None:
                try:
                    # ret, shared_frame.im[:] = self.cap.read()
                    # t = time.time()
                    ret, shared_frame.im = self.cap.read()
                    # ret, shared_frame.im[:] = self.cap.read()
                    # ret = self.cap.read(shared_frame.im[:])
                    # self.n += 1
                    # self.dtmean = (self.dtmean * (self.n-1) + time.time()-t) / self.n
                    # print('dt={}'.format(self.dtmean))
                except Exception as e:
                    pass
                if ret:
                    self.timestamp = self.get_timestamp()
                    shared_frame.timestamp = self.timestamp
                return ret
            else:
                # Return the next captured frame
                return self.cap.read(), self.get_timestamp()

    def close(self):
        try:
            self.cap.release()
        finally:
            self.cap = None

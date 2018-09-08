from __future__ import print_function
import skvideo.io
from opencv_grabber import *

# WRONG INTERFACE for skvideo! REWRITE! Switch to the generator vreader in skvideo

class FrameGrabberSkvideoFile(FrameGrabberCV2File):
    def gen_capture_obj(self):
        pass

    def open_file_for_capture(self, filename):
        cap = skvideo.io.FFmpegReader(filename)


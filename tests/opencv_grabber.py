""" Test whether the framegrabber can capture from the camera """
__package__ = 'imagemp'
from imagemp.shframe_grabbers.opencv_grabber import FrameGrabberCV2Camera

fg = FrameGrabberCV2Camera(0)
(success, im), timestamp = fg.capture()
print(success, ',', type(im), ',', timestamp)
fg.close()

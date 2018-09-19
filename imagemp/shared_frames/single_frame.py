from __future__ import print_function, division
from .shared_frame import SharedFrame
from .abstract_struct import SharedDataStructureAbstract
import ctypes

class SharedSingleFrame(SharedDataStructureAbstract):
    def __init__(self,
                 im_shape=(1, 1),
                 array_type=ctypes.c_uint16,
                 timestamp_type=ctypes.c_float,
                 iframe_type=ctypes.c_uint64,
                 lock=True):
        super(SharedSingleFrame, self).__init__()
        self.frame = SharedFrame(im_shape, array_type=array_type, timestamp_type=timestamp_type,
                                 iframe_type=iframe_type, lock=lock)

    def upload_new_element(self, im=None, timestamp=None):
        if im is not None:
            self.frame.im[:] = im     # just point to the same address as im
            if timestamp is not None:
                assert isinstance(timestamp, (int, long, float)), \
                    'Timestamp should be a number. Given: {}'.format(timestamp)
                self.frame.timestamp = timestamp
            else:
                self.frame.timestamp = self.timestamp

    def next_element2write2(self, update=None):
        return self.frame

    @property
    def last_written_element(self):
        return self.frame

    def element_following(self, element_id=None):
        return self.frame

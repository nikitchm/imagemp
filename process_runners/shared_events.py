from __future__ import print_function
import ctypes
import multiprocessing as mp
from imagemp.shared_frames.value_change_event import ValueChangeEvent

class SharedEvents(object):
    def __init__(self, timestamp_type=ctypes.c_float):
        self.capture_frame = mp.Event()       #
        self.frame_acquired = ValueChangeEvent(value_type=timestamp_type)
        self._mp_objects = []

    def add_mp_object(self, obj):
        self._mp_objects.append(obj)

    def exitall(self):
        for mp_object in self._mp_objects:
            try:
                mp_object.exit()
            except Exception as e:
                print(e)

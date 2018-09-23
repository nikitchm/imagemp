from __future__ import print_function
import ctypes
import multiprocessing as mp
# __package__ = 'imagemp'
from ..shared_frames.value_change_event import ValueChangeEvent


class SharedEvents(object):
    """ A class containing events and associated methods that should be shared between process-runner classes.
    In addition to providing a container for shared events, it provides methods: add_mp_object() and exitall(),
    which both emphasize the importance of the process runners to all have the .exit() method as a common interface
    for interrupting them, and provides a convenient way for doing it. """
    def __init__(self, timestamp_type=ctypes.c_float):
        self.capture_frame = mp.Event()  # an tragger event for a frame acquisition from a process
                                         # running in parallel to a frame grabber
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

from __future__ import print_function
import sys
# sys.path.append(r'/mnt/data/Dropbox/rnd/python/')
# import imagemp.add_parent_folder_to_path
import ctypes
import numpy as np
import multiprocessing as mp
import copy
import os
# print(sys.path)
from ..scheduler.scheduler import Time, Scheduler

degugging = False


def list_cum_mult(numbers):
    res = 1
    for i in numbers:
        res *= i
    return res


class SharedFrame(object):
    __ncalls = 0  # number of im updates in all instances of the class

    def __init__(self,
                 im_shape,
                 array_type=ctypes.c_uint16,
                 timestamp_type=ctypes.c_float,
                 iframe_type=ctypes.c_uint64,
                 lock=True):
        self.__im_shape = im_shape
        self.__nelem = list_cum_mult(im_shape)
        self.__array_type = array_type
        self.__timestamp_type = timestamp_type
        self.__iframe_type = iframe_type
        self.__lock = lock
        self.__array = None
        self.__timestamp = None
        self.__iframe = None
        self.init_mp_elements()
        self.__im = None
        self.id_orig_proc = os.getpid()

    def init_mp_elements(self):
        self.__array = mp.Array(self.__array_type, self.__nelem, lock=self.__lock)
        self.__timestamp = mp.Value(self.__timestamp_type, 0)
        self.__iframe = mp.Value(self.__iframe_type, 0)

    @property
    def timestamp(self):
        return self.__timestamp.value

    @timestamp.setter
    def timestamp(self, new_value):
        self.__timestamp.value = new_value

    @property
    def iframe(self):
        return self.__iframe.value

    @iframe.setter
    def iframe(self, newval):
        self.__iframe.value = newval

    def __init_im(self):
        # should be called only after calling the run() method in mp.Process inheriting class
        if isinstance(self.__array, mp.sharedctypes.SynchronizedArray):
            self.__im = np.ctypeslib.as_array(self.__array.get_obj())
        else:
            # e.g., when the Lock=False in self.array --> a.k.a. mp.RawArray
            self.__im = np.ctypeslib.as_array(self.__array)
        self.__im = self.__im.reshape(self.__im_shape)

    @property
    def im(self):
        """

        Returns
        -------
        mp.Array
        """
        if self.__im is None:
            self.__init_im()
        return self.__im

    @im.setter
    def im(self, new_im):
        if self.__im is None:
            self.__init_im()
        assert isinstance(new_im, np.ndarray)
        try:
            self.__im[:] = new_im   # copy data in new_im into the self.__im
        except Exception as e:
            print(e)
        self.increment_class_call_counter()
        self.iframe = self.__ncalls

    def get_all(self):
        """
        A convenience function. Return im, timestamp and __ncalls in one output

        Returns
        -------
        mp.Array(),
        mp.Value()
        int
        """
        return self.im, self.timestamp, self.iframe

    @classmethod
    def increment_class_call_counter(cls):
        cls.__ncalls += 1









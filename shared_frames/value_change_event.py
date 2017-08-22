import multiprocessing as mp
import ctypes

class ValueChangeEvent(object):
    # curval = MPValueProperty(value_type=ctypes.c_float, type_constructor_args=0)

    # Implements a multiprocessing event
    def __init__(self, value_type=ctypes.c_float):
        self.event1 = mp.Event()
        self.event2 = mp.Event()
        self.lock = mp.Lock()
        self.event1.set()
        self.__curval = mp.Value(value_type, -1)

    @property
    def curval(self):
        return self.__curval.value

    @curval.setter
    def curval(self, value):
        self.__curval.value = value

    def value_change(self, newval):
        with self.lock:
            if self.event1.is_set():
                self.event1.clear()
                self.event2.set()
            else:
                self.event2.clear()
                self.event1.set()
            self.curval = newval

    def get_event_to_wait(self):
        if self.event1.is_set():
            return self.event2
        else:
            return self.event1

    def wait_for_valchange(self, val, timeout=None):
        # print('val: {}, self.curval: {}'.format(val, self.curval))
        if val == self.curval:
            return self.get_event_to_wait().wait(timeout)
        else:
            return True

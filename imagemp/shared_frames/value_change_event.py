import multiprocessing as mp
import ctypes


# A multiprocessing 'switch' class. Provides an interface for the communication between processes
# when one process can change a shared value, and at the same time trigger an event indicating
# that thevalue was changed. Meanwhile, other processes can wait for the event to happen without
# running in a loop.
class ValueChangeEvent(object):
    # value = MPValueProperty(value_type=ctypes.c_float, type_constructor_args=0)

    # Implements a multiprocessing event
    def __init__(self, value_type=ctypes.c_float):
        self.__event1 = mp.Event()
        self.__event2 = mp.Event()
        self.__lock = mp.Lock()
        self.__event1.set()
        self.__curval = mp.Value(value_type, -1)

    @property
    def value(self):
        # Current value. Provides a convenience syntax and a potential to check the value for
        # correctness (in the setter; probably, most useful if this class is overloaded)
        return self.__curval.value

    @value.setter
    def value(self, value):
        self.__curval.value = value

    def value_change(self, newval):
        # Method to be called by the process changing the value. Triggers the switch of active events,
        # to be detected by the .wait_for_valchange() method
        with self.__lock:
            if self.__event1.is_set():
                self.__event1.clear()
                self.__event2.set()
            else:
                self.__event2.clear()
                self.__event1.set()
            self.value = newval

    def wait_for_valchange(self, val, timeout=None):
        # Pause the process until either the value is changed or the timeout is expired
        # print('val: {}, self.value: {}'.format(val, self.value))
        if val == self.value:
            return self.__get_event_to_wait().wait(timeout)
        else:
            return True

    def __get_event_to_wait(self):
        if self.__event1.is_set():
            return self.__event2
        else:
            return self.__event1

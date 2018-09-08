import ctypes

class MPValueProperty(object):
    # UNUSED at the moment
    # !The intent was to substitute the getters and setters for the mp.Value variables.
    # The innate limitation of the desrtiptors is that the variables of this class can only be
    # class variables, not instance ones. However, for some reason, even this didn't work with
    # multiprocessing.
    # https://www.smallsurething.com/python-descriptors-made-simple/
    # http://stackoverflow.com/questions/1004168/why-does-declaring-a-descriptor-class-in-the-init-function-break-the-descrip

    def __init__(self, value_type=ctypes.c_float, type_constructor_args=0):
        self.x = mp.Value(value_type, type_constructor_args)

    def __get__(self, obj, objtype):
        print('getting value : {}; type :{}'.format(self.x.value, type(self.x)))
        return self.x.value

    def __set__(self, obj, val):
        print('setting value to {}'.format(val))
        self.x.value = val

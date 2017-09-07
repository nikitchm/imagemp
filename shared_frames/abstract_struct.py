from shared_frame import SharedFrame
from ..scheduler.scheduler import Time

class SharedDataStructureAbstract(object):
    # This class exists exclusively as a guide for which methods and their signatures are expected
    # to be implemented in all SharedElements classes to be cross-compatible. Notice though that if
    # either of load_new_element and load_new_element is implemented the other can be omitted.
    def __init__(self):
        self.frame = SharedFrame((1,1)) # introduced here for code completion purpose only
        self.__clock = Time()

    # Copy the values from im into the existing array in a shared element in the class.
    def upload_new_element(self, im=None, timestamp=None):
        raise NotImplemented

    # Tell a writer which wants to share new data where it should be loaded (pointed to)
    def next_element2write2(self, update=True):
        return self.frame

    def next_element2write2_update_ref(self):
        pass

    # Return the element which was added last
    @property
    def last_written_element(self):
        return self.frame

    # Given element_id, return the element acquired immediately after it. If a consumer
    # is lagging behind the acquision by the shared data structure with multiple elements,
    # this function should provide a way to follow the acquired data sequentially, without
    # drops. Return the next element or None, if element_id is the last acquired.
    # The number of skipped elements in between (e.g., if lagging by more than the number of
    # elements in a cyclic data structure) can be obtained by taking the difference between
    # the element id's (0, if no lagging).
    def element_following(self, element_id=None):
        return self.frame

    # Return the current time timestamp
    @property
    def timestamp(self):
        return self.__clock.time()
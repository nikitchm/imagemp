from __future__ import print_function, division
from .shared_frame import SharedFrame
from .abstract_struct import SharedDataStructureAbstract
import multiprocessing as mp
import ctypes
import copy


class SharedFrameList(SharedDataStructureAbstract):
    # In this class the shared data structure is the list of SharedFrame elements which serves as
    # the circular buffer for the arriving data.
    def __init__(self,
                 im_shape=(1, 1),
                 nelem=1,
                 array_type=ctypes.c_uint16,
                 timestamp_type=ctypes.c_float,
                 iframe_type=ctypes.c_uint64,
                 lock=True,
                 overflow_di=2):
        super(SharedFrameList, self).__init__()
        self.nelem = nelem
        self.frames = []
        for i in range(nelem):
            self.frames.append(SharedFrame(im_shape, array_type=array_type, timestamp_type=timestamp_type,
                                           iframe_type=iframe_type, lock=lock))
        self.__i_last = mp.Value(ctypes.c_int32, -1)
        # self.__i_next_to_write = mp.Value(ctypes.c_int32, 0)
        self.overflow_di = overflow_di
        self.debugging = False

    @property
    def i_last(self):
        return self.__i_last.value

    @i_last.setter
    def i_last(self, val):
        self.__i_last.value = val

    @property
    def i_next_to_write(self):
        ilast = self.i_last
        return ilast + 1 if ilast + 1 < self.nelem else 0
        # return self.__i_next_to_write.value

    # @i_next_to_write.setter
    # def i_next_to_write(self, val):
    #     self.__i_next_to_write.value = val

    def upload_new_element(self, im=None, timestamp=None):
        if im is not None:
            frame_to_fill = self.next_element2write2()
            frame_to_fill.im[:] = im
            if timestamp is not None:
                assert isinstance(timestamp, (int, long, float)), \
                    'Timestamp should be a number. Given: {}'.format(timestamp)
                frame_to_fill.timestamp = timestamp
            else:
                frame_to_fill.timestamp = self.timestamp

    def next_element2write2(self, update=True):
        current_next = self.i_next_to_write
        # if update:
        #     self.i_last = current_next
        #     self.i_next_to_write = self.i_next_to_write + 1 if self.i_next_to_write + 1 < self.nelem else 0
        return self.frames[current_next]

    def next_element2write2_update_ref(self):
        self.i_last = self.i_next_to_write
        # print('i_last: {}'.format(self.i))
        # self.i_next_to_write = self.i_next_to_write + 1 if self.i_next_to_write + 1 < self.nelem else 0

    @property
    def last_written_element(self):
        return self.frames[self.i_last] if self.i_last >= 0 is not None else None

    def element_following(self, iframe2follow=None):
        # iframe_last = self.last_written_element.iframe
        iframe_last = copy.copy(self.last_written_element.iframe)
        if iframe_last is not None and iframe2follow < iframe_last:  # check if the new element is available
            if iframe_last - iframe2follow > self.nelem - self.overflow_di + 1:
                # following_ilist = (iframe_last - iframe2follow) % self.nelem
                following_ilist = (iframe_last + self.overflow_di) % self.nelem
                following_el = self.frames[following_ilist]
                if self.debugging: print('1: ', end='')
            else:
                following_ilist = (iframe2follow + 1) % self.nelem   # the index in the list
                following_el = self.frames[following_ilist]
                if self.debugging: print('2: ', end='')
            if self.debugging:
                iframes = [self.frames[0].iframe, self.frames[1].iframe, self.frames[2].iframe]
                min_iframe = min([self.frames[i].iframe for i in
                                  range(self.nelem) if iframe2follow<self.frames[i].iframe])
                min_iframes = [min_iframe, min_iframe+1]
                print('{}({}) --> {}[{}] ({}) : {}({}, {}) : '.format(
                    iframe2follow, following_el.iframe in min_iframes, following_el.iframe, following_ilist, iframe_last,
                    self.last_written_element.iframe, self.i_last, self.last_written_element.iframe == max(iframes)),
                    end='')
                for i in range(self.nelem):
                    print(self.frames[i].iframe, end=', ')
                print()
        else:
            following_el = None
        return following_el

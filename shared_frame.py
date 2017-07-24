from __future__ import print_function
import sys
# sys.path.append(r'/mnt/data/Dropbox/rnd/python/')
import add_parent_folder_to_path
import ctypes
import numpy as np
import multiprocessing as mp
import time
import copy
import os
# print(sys.path)
from common.scheduler import Time, Scheduler

degugging = False

# TODO: Introduce a frame identifiers (idr) which the frame grabber would pass with the frame
# and using which, when received as an input, be able to position among other frames.
# TODO: Enable the scheduler to allow for acquisition / processing of the frames in time at
# predetermined time moments.
# TODO: Allow the processor to drive the acquisition (as part of the scheduler?)

# the following is taken from movies3 file
def get_opencv_prop(obj, prop):
    import cv2
    """ Get the propery, prop, of the opencv opject, obj.
    Works for both, Python 2 and 3+ (that's the main reason for introducing this interface).
     """

    # print('params: {}'.format(prop))
    if int(cv2.__version__.split('.')[0]) < 3:
        val = obj.get(getattr(cv2.cv, 'CV_' + prop))
    else:
        val = obj.get(getattr(cv2, prop))
    return val


def list_cum_mult(numbers):
    res = 1
    for i in numbers:
        res *= i
    return res


class MPValueProperty(object):
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


class SharedSingleFrame(SharedDataStructureAbstract):
    def __init__(self, im_shape=(1, 1),
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
                if degugging: print('1: ', end='')
            else:
                following_ilist = (iframe2follow + 1) % self.nelem   # the index in the list
                following_el = self.frames[following_ilist]
                if degugging: print('2: ', end='')
            if degugging:
                iframes = [self.frames[0].iframe, self.frames[1].iframe, self.frames[2].iframe]
                min_iframe = min([self.frames[i].iframe for i in range(self.nelem) if iframe2follow<self.frames[i].iframe])
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


class FrameGrabberAbstract(object):
    def __init__(self):
        pass

    def capture(self, shared_frame):
        # Should update the shared frame (image and timestamp)
        pass

    def close(self):
        pass


class FrameGrabberCV2VideoReader(FrameGrabberAbstract):
    # n = 0
    # dtmean = 0

    def __init__(self, vid_filename, timestamps_src=None):
        super(FrameGrabberCV2VideoReader, self).__init__()
        self.filename = vid_filename
        self.timestamps_src = timestamps_src
        self.timestamps_method = self.init_timestamp_method()
        # self.cap = cv2.VideoCapture(self.filename)
        self.cap = get_opencv_kind_videoCapture(self.filename)
        self.timestamp = -1
        self.iframe = None

    def get_cv2_timestamp(self, obj):
        return get_opencv_prop(self.cap, 'CAP_PROP_POS_MSEC')

    def init_timestamp_method(self):
        # Return a method for getting the timestamp for the given frame
        if self.timestamps_src is None:
            return self.get_cv2_timestamp
        else:
            from functools import partial
            timestamp_func = None
            exec ('timestamp_func = {}'.format(self.timestamps_src['timestamp_func']))
            # assert isinstance(self.timestamps_src, dict), 'timestamps_src should be either None or a dictionary'
            return partial(timestamp_func, **self.timestamps_src['kwargs'])

    def capture(self, shared_frame):
        ret = False
        if self.cap.isOpened():
            try:
                # ret, shared_frame.im[:] = self.cap.read()
                # t = time.time()
                ret, shared_frame.im = self.cap.read()
                # ret, shared_frame.im[:] = self.cap.read()
                # ret = self.cap.read(shared_frame.im[:])
                # self.n += 1
                # self.dtmean = (self.dtmean * (self.n-1) + time.time()-t) / self.n
                # print('dt={}'.format(self.dtmean))
            except Exception as e:
                pass
            if ret:
                self.timestamp = self.timestamps_method(self)
                shared_frame.timestamp = self.timestamp
        return ret

    def close(self):
        self.cap.release()


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


class ProcessRunnerAbstract(mp.Process):
    def __init__(self):
        super(ProcessRunnerAbstract, self).__init__()
        self.exit_event = mp.Event()

    def exit(self):
        # Triggers an exit event which should be detected in the process loop
        self.exit_event.set()

    def is_exiting(self):
        return self.exit_event.is_set()


class FrameGrabberRunner(ProcessRunnerAbstract):
    # Present an interface for running different kinds of frame grabbers and
    # schedulers in a separate process
    def __init__(self, shared_data=SharedDataStructureAbstract(), grabber_settings=None,
                 shared_events=SharedEvents(), scheduler=Scheduler(dt=0)):
        super(FrameGrabberRunner, self).__init__()
        self.shared_data = shared_data
        self.grabber_settings = grabber_settings
        self.shared_events = shared_events
        self.scheduler = scheduler
        self.grabber = None     # to be determined in self.run()
        self._capture_frame_update_timeout = .1  # sets the minimal frequency of doing a while-loop iteration,
                                                 # when the capturing is triggered by the capture_frame event

    def run(self):
        try:
            # Past this point, the parent and other processes can communicate
            # with this class only via inter-process messages.
            # Initialize the grabber instance:
            print('FrameGrabberRunner: initializing the grabber')
            grabber_class = eval('{}'.format(self.grabber_settings['class']))
            self.grabber = grabber_class(**self.grabber_settings['kwargs'])
            print('FrameGrabberRunner: running')
            if self.scheduler is not None:
                self.scheduler.run_sequence(func=self.capture, run_in_new_thread=True)
                # The scheduler now runs in a different thread, and in order to pause here
                # until the exit_event is generated, it's necessary to wait for it explicitely
                self.exit_event.wait()
            else:
                # If no scheduler is set up, capture images only when capture_frame event is triggered.
                while not self.is_exiting():
                    ret = self.shared_events.capture_frame.wait(self._capture_frame_update_timeout)
                    if ret:
                        self.capture()

        finally:
            print('FrameGrabberRunner : stopping...')
            if self.scheduler is not None:
                self.scheduler.stop()
            self.grabber.close()

    def capture(self):
        # capture into the shared element referred to by .next_element2write2()
        # next_shared_frame2write2 = self.shared_data.next_element2write2()
        self.shared_data.next_element2write2_update_ref()
        # ret = self.grabber.capture(next_shasred_frame2write2)
        ret = self.grabber.capture(self.shared_data.next_element2write2())
        if ret:
            # update shared_data.i_last
            # self.shared_data.next_element2write2_update_ref()
            self.shared_events.frame_acquired.value_change(self.shared_data.last_written_element.timestamp)
        # t2 = time.time()
        # print('capturing fps: {}'.format(int(round(1/(t2-t1)))))
        # t1 = t2


import cv2
class SimpleDisplay(ProcessRunnerAbstract):
    ilast = 0

    def __init__(self, shared_data=SharedDataStructureAbstract(), shared_events=SharedEvents(),
                 scheduler=Scheduler()):
        super(SimpleDisplay, self).__init__()
        self.shared_data = shared_data
        self.shared_events = shared_events
        self.scheduler = scheduler
        self.last_timestamp = self.shared_events.frame_acquired.curval
        self.__timestamp_update_timeout = 1  # s
        self.nframes_shown = 0

    def run(self):
        try:
            # display the last acquired frame
            tprev = None
            while not self.is_exiting():
                # Avoid polling -- wait for the event indicating that a new frame has been acquired
                frame_acquired = self.shared_events.frame_acquired.wait_for_valchange(self.last_timestamp,
                                                                                      self.__timestamp_update_timeout)
                if frame_acquired:   # if didn't timeout
                    t = time.time()
                    fps = int(1/(t - tprev)) if tprev else None
                    tprev = t
                    self.nframes_shown += 1
                    im = self.shared_data.last_written_element.im.astype('uint8')
                    self.last_timestamp = self.shared_data.last_written_element.timestamp
                    cv2.putText(im, 'frame#: {}, fps: {}, d(iframe): {}'.
                                format(int(self.last_timestamp),
                                       fps,
                                       self.shared_data.last_written_element.iframe-self.ilast),
                                (15, 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    self.ilast = self.shared_data.last_written_element.iframe
                    cv2.imshow('im', im)
                    keypressed = cv2.waitKey(1)
                    if keypressed in [81, 113]:   # 'Q' or 'q' : exit
                        self.exit()
        except Exception as e:
            print('Exception: {}'.format(e))
        finally:
            cv2.destroyAllWindows()


class Recorder(ProcessRunnerAbstract):
    def __init__(self, filename, shared_data=SharedDataStructureAbstract(), shared_events=SharedEvents(),
                 rec_opts=None, scheduler=Scheduler()):
        super(Recorder, self).__init__()
        self.shared_data = shared_data
        self.shared_events = shared_events
        self.scheduler = scheduler
        self.last_timestamp = self.shared_events.frame_acquired.curval
        self.__timestamp_update_timeout = 1  # s
        self._i_last_element = 0
        self.filename = filename
        self.recorder = None
        self.set_rec_opts()

    def set_rec_opts(self, fourcc='XVID', is_color=True, shape=(100,100), fps=30):
        self.fourcc = fourcc
        self.is_color = is_color
        self.fps = fps
        self.shape = shape

    def run(self):
        try:
            print('Started recording...')
            cv2_fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
            self.recorder = cv2.VideoWriter()
            ret = self.recorder.open(self.filename, cv2_fourcc, self.fps, self.shape, isColor=True)
            if not ret: exit(1)

            while not self.is_exiting():
                frame_acquired = self.shared_events.frame_acquired.wait_for_valchange(self.last_timestamp,
                                                                                self.__timestamp_update_timeout)
                if frame_acquired:  # if didn't timeout
                    # Note that if the frame access isn't locked the values of im and timestamp
                    # can change while being read. Using a data structure with more than one element
                    # would help, as well as locking the access to the elements while writing into them.
                    try:
                        im, self.last_timestamp, self._i_last_element =\
                            self.shared_data.element_following(self._i_last_element).get_all()
                        # im, self.last_timestamp, self._i_last_element = self.shared_data.last_written_element.get_all()
                        im = im.astype('uint8')
                        self.recorder.write(im)
                    except AttributeError:
                        pass
                    except Exception as e:
                        print('Recorder Exception: {}, {}'.format(type(e).__name__, e.args))
        except Exception as e:
            print('RECORDER ERROR: {}'.format(e))
        finally:
            self.recorder.release()
            print('Stopped recording.')


class Analysis(ProcessRunnerAbstract):
    def __init__(self, vid_av_filename, shared_data=SharedDataStructureAbstract(), shared_events=SharedEvents(),
                 scheduler=Scheduler()):
        super(Analysis, self).__init__()
        self.shared_data = shared_data
        self.shared_events = shared_events
        self.scheduler = scheduler
        self.last_timestamp = self.shared_events.frame_acquired.curval
        self.__timestamp_update_timeout = 1  # s
        self._i_last_element = 0
        self.vid_av_filename = vid_av_filename
        self.fish_tracker = None
        self.im_av = np.empty((2,2))
        self.fish_tracker_init = False

    def run(self):
        try:
            self.fish_tracker_method(None, None)
            print('Started analysis...')
            while not self.is_exiting():
                frame_acquired = self.shared_events.frame_acquired.wait_for_valchange(self.last_timestamp,
                                                                                self.__timestamp_update_timeout)
                if frame_acquired:  # if didn't timeout
                    # Note that if the frame access isn't locked the values of im and timestamp
                    # can change while being read. Using a data structure with more than one element
                    # would help, as well as locking the access to the elements while writing into them.
                    try:
                        im, self.last_timestamp, self._i_last_element =\
                            self.shared_data.element_following(self._i_last_element).get_all()
                        # im, self.last_timestamp, self._i_last_element = self.shared_data.last_written_element.get_all()
                        im = im.astype('uint8')
                        t0 = time.time()
                        self.fish_tracker_method(im, self.last_timestamp)
                        print('processed: ', self._i_last_element, ' dt: ', time.time() - t0)
                    except AttributeError:
                        pass
                    except Exception as e:
                        print('Analysis Exception: {}, {}'.format(type(e).__name__, e.args))
        except Exception as e:
            print('ANALYSIS ERROR: {}'.format(e))
        finally:
            print('Stopped analysis.')

    # @staticmethod
    def fish_tracker_method(self, im, t):
        if self.fish_tracker_init:
            self.fish_tracker.process_frame(im)
        else:
            from analysis.behavior.tracker.findfish4class import FindFish4
            self.im_av = np.load(self.vid_av_filename)
            self.fish_tracker = FindFish4(im_av=self.im_av)
            self.fish_tracker_init = True


def get_file_timestamp(obj, filename=''):
    obj.iframe = get_opencv_prop(obj.cap, 'CAP_PROP_POS_FRAMES')
    obj.iframe = int(obj.iframe)
    return obj.iframe

def get_opencv_kind_videoCapture(filename):
    import cv2
    cap = cv2.VideoCapture()
    c = cap.open(filename=filename)
    if not c:
        try:
            cap.release()
            import skvideo.io
            cap = skvideo.io.VideoCapture(filename)
        except Exception as e:
            print("Couldn't open {}".format(filename))
    return cap

if __name__ == '__main__':
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="path to input video file")
    ap.add_argument("-r", "--recvid", required=False, help="path to recorded video file")
    args = vars(ap.parse_args())
    vid_filename = args['video'] if args['video'] is not None else \
        r'C:\data\rnd\Max_Experiments\behave\MTZCont_Optovin_fish5.vid.avi'
        # r'C:\data\rnd\Max_Experiments\behave\MTZcont_Optovin_fish5.vid.stimLED.avi'
        # r'/mnt/data/rnd/LargeData/MTZCont_Optovin_fish5.vid.avi'
    vid_av_filename = os.path.splitext(vid_filename)[0] + '_average.npy'

    # Define grabber parameters:
    cap = get_opencv_kind_videoCapture(vid_filename)
    # cap = cv2.VideoCapture(vid_filename)
    shape = cap.read()[1].shape
    cap.release()
    # shape = (480, 480, 3)

    timestamps_src = {'timestamp_func': 'get_file_timestamp',
                      'kwargs': {'filename': 'timestamps.txt'}}
    # timestamps_src = None
    grabber_settings = {'class': 'FrameGrabberCV2VideoReader',
                        'kwargs': {'vid_filename': vid_filename,
                                   'timestamps_src': timestamps_src}}
    timestamp_type = ctypes.c_float
    # Initialize shared data structure
    if 0:
        print('Using SharedSingleFrame')
        shared_d = SharedSingleFrame(im_shape=shape,
                                     array_type=ctypes.c_uint16,
                                     timestamp_type=timestamp_type,
                                     lock=False)
    else:
        print('Using SharedFrameList')
        shared_d = SharedFrameList(im_shape=shape,
                                   nelem=100,
                                   array_type=ctypes.c_uint16,
                                   timestamp_type=timestamp_type,
                                   lock=False)
    # Create shared_events (can also be taken from FrameGrabber after its initialization)
    shared_events = SharedEvents()

    # framegrabber scheduler
    fg_scheduler = Scheduler(dt=0.05)
    # Start the framegrabber
    frame_grabber = FrameGrabberRunner(shared_data=shared_d,
                                       grabber_settings=grabber_settings,
                                       shared_events=shared_events,
                                       scheduler=fg_scheduler)

    # Start a display
    display = SimpleDisplay(shared_data=shared_d,
                            shared_events=shared_events)

    # Start the recorder
    vfsplit = os.path.splitext(vid_filename)
    vid_rec_filename = vfsplit[0] + '_rec' + vfsplit[1]
    recorder = Recorder(filename=vid_rec_filename,
                        shared_data=shared_d,
                        shared_events=shared_events)
    recorder.set_rec_opts(fourcc='XVID', is_color=True, shape=(480,480), fps=30)

    # Start the analysis
    analysis = Analysis(vid_av_filename,
                        shared_data=shared_d,
                        shared_events=shared_events)

    display.start()
    recorder.start()
    analysis.start()
    time.sleep(5)
    frame_grabber.start()

    for obj in [frame_grabber, display, recorder, analysis]:
        shared_events.add_mp_object(obj)
    # for obj in [frame_grabber, display, recorder]:
    #     shared_events.add_mp_object(obj)

    t2run = 10
    time.sleep(t2run)
    print('{} sec have expired. Exiting.'.format(t2run))
    shared_events.exitall()

from __future__ import print_function
import ctypes
import time
import numpy as np
from .scheduler.scheduler import Scheduler
from .shared_frames.shared_frame import *
from .shframe_grabbers.factory import get_grabber
from abstract import ProcessRunnerAbstract
degugging = False
from ..shared_frames.abstract_struct import SharedDataStructureAbstract


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
                    self.ilast = self.shared_data.last_written_element.iframe
                    cv2.putText(im, 'frame#: {}, fps: {}, d(iframe): {}'.
                                format(int(self.last_timestamp),
                                       fps,
                                       self.shared_data.last_written_element.iframe-self.ilast),
                                (15, 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    cv2.imshow('im', im)
                    keypressed = cv2.waitKey(1)
                    if keypressed in [81, 113]:   # 'Q' or 'q' : exit
                        self.exit()
        except Exception as e:
            print('Exception: {}'.format(e))
        finally:
            cv2.destroyAllWindows()


class Recorder(ProcessRunnerAbstract):
    def __init__(self, filename, shared_data=SharedDataStructureAbstract(),shared_events=SharedEvents(),
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

    def set_rec_opts(self, fourcc='XVID', is_color=True, shape=(100,100), fps=30):
        self.fourcc = fourcc
        self.is_color = is_color
        self.fps = fps
        self.shape = shape

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
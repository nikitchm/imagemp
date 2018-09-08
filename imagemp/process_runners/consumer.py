from ..scheduler.scheduler import Scheduler
from ..shared_frames.shared_frame import SharedFrame
from ..shared_frames.abstract_struct import SharedDataStructureAbstract
from shared_events import SharedEvents
from abstract import ProcessRunnerAbstract
import numpy as np

class Consumer(ProcessRunnerAbstract):
    def __init__(self, shared_data=SharedDataStructureAbstract(),shared_events=SharedEvents(),
                 scheduler=Scheduler(), *args, **kwargs):
        super(Consumer, self).__init__()
        self.class_name = self.__class__.__name__
        self.shared_data = shared_data
        self.shared_events = shared_events
        self.scheduler = scheduler
        self.last_timestamp = self.shared_events.frame_acquired.curval
        self.im = np.empty((2,2))
        self._shared_frame = SharedFrame((0,0))
        self.next_to_acquire = 'last'   # 'last', 'next', ...
        self.__newval_timeout = 1  # s
        self.iframe = 0
        self.init_custom(*args, **kwargs)

    def run(self):
        try:
            self.log('Started {}...'.format(self.class_name))
            self.run_pre_loop()
            while not self.is_exiting():
                frame_acquired = \
                    self.shared_events.frame_acquired.wait_for_valchange(self.last_timestamp, self.__newval_timeout)
                if frame_acquired:  # if didn't timeout
                    # Note that if the frame access isn't locked the values of im and timestamp
                    # can change while being read. Using a data structure with more than one element
                    # would help, as well as locking the access to the elements while writing into them.
                    try:
                        self.run_loop_pre_acquire()
                        self.run_loop_acquire()
                        self.run_loop_post_acquire()
                    except AttributeError:
                        pass
                    except Exception as e:
                        print('{} Exception: {}, {}'.format(self.class_name, type(e).__name__, e.args))
        except Exception as e:
            print('{} ERROR: {}'.format(self.class_name.upper(), e))
        finally:
            self.close()
            self.log('{} stopped.'.format(self.class_name))

    def run_loop_pre_acquire(self):
        pass

    def init_custom(self, *args, **kwargs):
        pass

    def run_pre_loop(self):
        pass

    def run_loop_acquire(self):
        if self.next_to_acquire == 'next':
            self._shared_frame = self.shared_data.element_following(self.iframe)
        elif self.next_to_acquire == 'last':
            self._shared_frame = self.shared_data.last_written_element
        else:
            raise NotImplemented
        self.im, self.last_timestamp, self.iframe = self._shared_frame.get_all()

    def run_loop_post_acquire(self):
        pass

    def close(self):
        pass

    def log(self, mess):
        print(mess)

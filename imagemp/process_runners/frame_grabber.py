from __future__ import print_function, division
from .abstract import ProcessRunnerAbstract
from ..shared_frames.abstract_struct import SharedDataStructureAbstract
from ..shframe_grabbers.abstract import FrameGrabberAbstract
from .shared_events import SharedEvents
from ..scheduler.scheduler import Scheduler


class FrameGrabberRunner(ProcessRunnerAbstract):
    """ Presents an interface for running different kinds of frame grabbers and
    schedulers in separate processes """
    def __init__(self, shared_data=SharedDataStructureAbstract(), grabber_settings=None,
                 shared_events=SharedEvents(), scheduler=Scheduler(dt=0), grabber=FrameGrabberAbstract()):
        super(FrameGrabberRunner, self).__init__()
        self.shared_data = shared_data
        self.grabber_settings = grabber_settings
        self.shared_events = shared_events
        self.scheduler = scheduler
        self.grabber = grabber
        self._capture_frame_update_timeout = .1  # sets the minimal frequency of doing a while-loop iteration,
                                                 # when the capturing is triggered by the capture_frame event

    def run(self):
        try:
            # Past this point, the parent and other processes can communicate
            # with this class only via inter-process messages.
            # Start grabber components which are not pickable, s.a. VideoCapture() in opencv
            self.grabber.init_unpickable()
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
        # Capture into the shared element referred to by .next_element2write2()
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

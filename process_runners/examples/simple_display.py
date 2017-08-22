from ..consumer import Consumer
from ..timers import ElapsedTimer
import cv2

class SimpleDisplay(Consumer):
    def init_custom(self, *args, **kwargs):
        self.next_to_acquire = 'last'
        self.nframes_shown = 0
        self.t_acq = ElapsedTimer(moving_ave_n=10)
        self.t_fps = ElapsedTimer(moving_ave_n=10)
        self.t_fps.tic()

    def run_pre_loop(self):
        pass

    def run_loop_pre_acquire(self):
        self.nframes_shown += 1
        self.t_acq.tic()

    def run_loop_post_acquire(self):
        self.t_acq.toc()
        self.t_fps.toc()
        self.t_fps.tic()
        self.im = self.im.astype('uint8')
        cv2.putText(self.im, 'frame#:   {}, update fps: {}, d(iframe): {}'.
                    format(int(self.last_timestamp),
                           round(10 / (self.t_fps.dts.moving_average+1e-10))/10,
                           self.shared_data.last_written_element.iframe - self.iframe),
                    (15, 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow('im', self.im)
        keypressed = cv2.waitKey(1)
        if keypressed in [81, 113]:  # 'Q' or 'q' : exit
            self.exit()

    def close(self):
        cv2.destroyAllWindows()

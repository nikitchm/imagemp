from ..consumer import Consumer
import numpy as np
from analysis.behavior.tracker.findfish4class import FindFish4


class FishTracker4(Consumer):
    def init_custom(self, *args, **kwargs):
        self.next_to_acquire = 'next'
        self.vid_av_filename = kwargs['vid_av_filename'] if 'vid_av_filename' in kwargs else None
        self.im_av = np.load(self.vid_av_filename)
        self.fish_tracker = FindFish4(im_av=self.im_av)

    def run_pre_loop(self):
        pass

    def run_loop_pre_acquire(self):
        pass

    def run_loop_post_acquire(self):
        self.fish_tracker.process_frame(new_im=self.im.astype('uint8'),
                                        timestamp=self.last_timestamp)

    def close(self):
        pass

# Measure time using start/time methods for timers in a dictionary
# compute fps
# Measure statistics if necessary
# .time() method here is implemented via the standard time library
# and is meant to be overloaded if necessary (e.g., using more accurate methods)
import time


class Timer(object):
    def __init__(self, moving_ave_n=10):
        self.n = 0
        self.tprev = None
        self.tlast = None
        self.dt = None
        self.fps = None
        self.compute_moving_ave = moving_ave_n > 0
        self.dt_cycl_buf = MovingAverageList(moving_ave_n)

    def time(self):
        return time.time()

    def newtime(self):
        self.tprev = self.tlast
        self.tlast = self.time()
        self.dt = self.tlast - self.tprev
        if self.compute_moving_ave:
            self.dt_cycl_buf.update(self.dt)


class ElapsedTimer(object):
    def __init__(self, moving_ave_n=10):
        self.n = 0
        self.t0 = None
        self.t1 = None
        self.dt_last = None
        self.fps = None
        self.compute_moving_ave = moving_ave_n > 0
        self.dts = MovingAverageList(moving_ave_n)
        self.active = True  # allows deactivating the timer with minimal modification to the parent code

    def time(self):
        return time.time()

    def tic(self):
        if self.active:
            self.t0 = self.time()

    def toc(self):
        if self.active:
            self.t1 = self.time()
            self.dt_last = self.t1 - self.t0
            if self.compute_moving_ave:
                self.dts.update(self.dt_last)
            return self.dt_last, self.dts.moving_average
        else:
            return None, None

    def set_active_state(self, bool_state):
        self.active = bool_state


class MovingAverageList(object):
    """
    An efficient python implementation of the moving average computation
    """
    # TODO: allow computation based on a provided (and externally updated) list
    def __init__(self, nel):
        self.nel = nel
        self.list = []
        self.ilast = -1
        self.is_filled = False
        self.sum = 0.0
        self.moving_average = None

    def update(self, newel):
        if self.is_filled:
            self.ilast = (self.ilast + 1) % self.nel
            self.sum = self.sum - self.list[self.ilast] + newel
            self.list[self.ilast] = newel
            self.moving_average = self.sum / self.nel
        else:
            self.sum += newel
            self.list.append(newel)
            self.ilast += 1
            if self.ilast == self.nel - 1:
                self.is_filled = True
            self.moving_average = self.sum / (self.ilast+1)
        return self.moving_average

    def get_ordered_list(self):
        # return the list in order of addition to the list
        return self.list[self.ilast+1:] + self.list[:self.ilast+1]


if __name__ == '__main__':
    from random import random as rand
    et = ElapsedTimer(moving_ave_n=10)

    sum = 0
    n = 10
    for i in range(n):
        et.tic()
        t = .5+rand()/10
        sum += t
        time.sleep(t)
        print(et.toc())
    print('mean: {}'.format(sum/n))

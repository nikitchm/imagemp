import threading

def print_func(message):
    print(message)


def pause(dt, finished_event=threading.Event()):
    finished_event.wait(dt)
    finished_event.clear()


class TimeAbstract(object):
    # The abstract time class specifying the methods that a class provided to the scheduler
    # to get the time reference should possess.
    def __init__(self):
        pass

    def time(self):
        pass


class Time(TimeAbstract):
    """ This class provides .time() method to get the current time in seconds. """
    import time as time_module

    def __init__(self):
        super(Time, self).__init__()
        pass

    def time(self):
        return self.time_module.time()

    def tuple_to_sec_since_epoch(self, tup):
        # E.g., tuple(time.localtime()) --> (2017, 3, 1, 7, 53, 35, 2, 60, 0) --> tup --> 1488372815.0
        return self.time_module.mktime(tup)


class _DtScheduler(threading.Thread):
    """
    A helper class introduced to allow for shared interface for running a sequence of function calls
    in the same thread as the caller and a new one.
    """
    def __init__(self, func, tseq, n, interrupt_event, t_func, logger, f_args, f_kwargs):
        threading.Thread.__init__(self)
        self.n = n
        self.tseq = tseq
        self.func = func
        self.f_args = f_args
        self.f_kwargs = f_kwargs
        self.logger = logger
        self.interrupt_event = interrupt_event
        self.dt_finished = threading.Event()    # An event used to pause the processing
        self.t = t_func
        self.setDaemon(True)

    def run(self):
        i = 0
        while True:
            if self.interrupt_event is not None and self.interrupt_event.is_set():
                self.logger('scheduler:repeat_every_dt : timer has stopped. Breaking the loop.')
                break
            if self.n is not None and i > self.n:
                self.logger('scheduler:repeat_every_dt : Exceeded the number of iterations ({}/{}). Breaking the loop.'.
                            format(i, self.n))
                break
            else:
                i += 1
            tnext = next(self.tseq)

            self.dt_finished.wait(tnext - self.t.time())
            if not self.dt_finished.is_set():   # if expired (hasn't been canceled)
                self.func(*self.f_args, **self.f_kwargs)
            self.dt_finished.clear()

    def cancel(self):
        self.dt_finished.set()


class Scheduler(object):
    def __init__(self, func=None, dt=None, t0=-1.1, n=-1, t_func=None, logger=print_func,
                 run_in_new_thread=True, f_args=None, f_kwargs=None):
        self._t = Time()
        self._timer = None
        self._func = None
        self._t0 = None
        self._dt = None
        self._n = None
        self._run_in_new_thread = None
        self._f_args = tuple()
        self._f_kwargs = dict()
        self._logger = logger
        self.__finished = None
        self.update_attributes(func=func, dt=dt, t0=t0, n=n, t_func=t_func,
                               run_in_new_thread=run_in_new_thread, f_args=f_args, f_kwargs=f_kwargs)

    @property
    def _finished(self):
        # Allows to avoid defining self.__finished = threading.Event() in the __init__(),
        # which prevents using the scheduler along with the multiprocessing module as
        # the threading.lock isn't pickable
        if self.__finished is None:
            self.__finished = threading.Event()
        return self.__finished

    def update_attributes(self, func=None, dt=None, t0=-1.1, n=-1, t_func=None,
                          run_in_new_thread=None, f_args=None, f_kwargs=None):
        """
        Update the class attributes. If the parameters are at their default values (e.g., omitted
        in the call to this method), they are kept at their's current value.
        Parameters
        ----------
        func : None or function
        dt : None or iterable or generator or number
        t0 : None or number
        n : None or number
        t_func : None or class instance
        run_in_new_thread : bool
        f_args : args
        f_kwargs : kwargs
        """
        # Update attributes only if new (not equal argument defaults), valid values are provided
        self._func = func if func is not None else self._func
        self._dt = dt if dt is not None else self._dt
        self._t0 = t0 if t0 is None or t0 != -1.1 else self._t0
        self._n = n if n is None or n >= 0 else self._n
        self._t = t_func if t_func is not None else self._t
        self._run_in_new_thread = run_in_new_thread if run_in_new_thread is not None else self._run_in_new_thread
        self._f_args = f_args if f_args is not None else self._f_args
        self._f_kwargs = f_kwargs if f_kwargs is not None else self._f_kwargs
        return self

    @property
    def t_now(self):
        return self._t.time()

    def start_at(self, func, t, *f_args, **f_kwargs):
        """ Start func(*f_args, **f_kwargs) at t (in the format of Time().time().
        E.g., t=Time.time() + 100.5 -- start in 100.5 sec, or
        t=Time.tuple_to_sec_since_epoch((2017,3,1,18,0,0,0,0,0)) """
        self.start_in(func, t - self.t_now, *f_args, **f_kwargs)

    def start_in(self, func, dt, *f_args, **f_kwargs):
        """ Start func(*f_args, **f_kwargs) after dt seconds. """
        dt = max(0, dt)
        if self._run_in_new_thread:
            self._timer = threading._Timer(dt, func, args=f_args, kwargs=f_kwargs)
            self._timer.start()
        else:
            self._finished.wait(dt)
            if not self._finished.is_set():
                func(*f_args, **f_kwargs)

    @staticmethod
    def tseq_gen(t0, dt):
        """ Sequence generator, starting at t0 and either adding dt
        at each iteration, if dt is a number, or adding next value in
        dt, if it's an iterable.
        """
        def dt_gen(x):
            is_iter = hasattr(x, '__iter__')
            if is_iter:
                x = iter(x)
            while True:
                if is_iter:
                    yield next(x)
                else:
                    yield x
        t = t0
        dt = dt_gen(dt)
        while True:
            yield t
            try:
                t += next(dt)
            except StopIteration:
                break
        yield (None)

    def run_sequence(self, func=None, dt=None, t0=-1.1, n=-1, run_in_new_thread=None, f_args=None, f_kwargs=None):
        """
        Execute func(*f_args, **f_kwargs) at times separated by dt seconds. dt can be a
        number or an iterable. The number of iterations is set either by n, if not None,
        or by the length of dt, if it's finite. Otherwise, the iterations continue until
        the scheduler's method .stop() is called. Iterations start at t0, if not None,
        otherwise, immediately.

        The parameters can be set either at the time of the function call, or during the
        class instance construction, or using the update_attributes method.

        Parameters
        ----------
        func : function
        dt : number or iterable
            seconds
        t0 : None or number
            Number corresponding to the Time().time() format
        n : None or number
        run_in_new_thread : bool
        f_args : None or tuple
        f_kwargs : None or dict
        """
        self.update_attributes(func=func, dt=dt, t0=t0, n=n, run_in_new_thread=run_in_new_thread,
                               f_args=f_args, f_kwargs=f_kwargs)
        if self._func is None or self._dt is None:
            raise ValueError("Scheduler:repeat_every_dt : either func, or dt is not provided.")

        # form the sequence generator
        self._t0 = self._t0 or self.t_now
        if self._n is None and hasattr(self._dt, '__len__'):
            self._n = len(self._dt)
        tseq = self.tseq_gen(self._t0, self._dt)

        self._finished.clear()
        self._timer = _DtScheduler(self._func, tseq, self._n, interrupt_event=self._finished, t_func=self._t,
                                   logger=self._logger, f_args=self._f_args, f_kwargs=self._f_kwargs)
        if self._run_in_new_thread:
            self._timer.start()
        else:
            self._timer.run()

    def join(self):
        """
        If a method is running in a new thread, wait for it's completion to continue.
        """
        if self._run_in_new_thread:
            self._timer.join()

    def stop(self):
        if self._timer is not None:  # if it ever ran
            self._timer.cancel()


if __name__ == '__main__':
    import time

    def test(text='tests: '):
        print(text, Time().time())

    sch = Scheduler()

    print('init time: {}'.format(Time().time()))
    # repeat_every_dt allows running the function on at scheduled intervals,
    # which can be set as an iterable/generator, or a number, in which case the iterations repeat until
    # a stopping signal is received.
    # Below --> 4 executions: in 1, 2, 2.5, 2.7 seconds after now
    sch.run_sequence(test, dt=[1, .5, .2], t0=Time().time() + 1, f_kwargs={'text': 'tests='}, run_in_new_thread=True)
    print('Waiting for the repeations to end...')
    sch.join()  # if running in another thread, wait until the iterations are completed

    print('\nStarting endless calling...')
    # The class attributes can be updated separately from the repeat_every_dt method call.
    sch.update_attributes(func=test, t0=None, n=None)
    sch.run_sequence(dt=0.5, run_in_new_thread=True)
    print('sleeping in the main thread for 3 seconds and stopping the repeatitions after that.')
    time.sleep(3)
    sch.stop()
    print('The repeatitions are stopped.')

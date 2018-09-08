import multiprocessing as mp

class ProcessRunnerAbstract(mp.Process):
    def __init__(self):
        super(ProcessRunnerAbstract, self).__init__()
        self.exit_event = mp.Event()

    def exit(self):
        # Triggers an exit event which should be detected in the process loop
        self.exit_event.set()

    def is_exiting(self):
        return self.exit_event.is_set()
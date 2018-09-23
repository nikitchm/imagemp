import multiprocessing as mp


# Define default set of methods to be present in process runners (e.g., a consumer or frame-grabber).
# Specify the termination methods (exit() and is_exiting()) which would allow the interruption of the
# process running from another process
class ProcessRunnerAbstract(mp.Process):
    def __init__(self):
        super(ProcessRunnerAbstract, self).__init__()
        self.exit_event = mp.Event()

    def run(self):
        pass

    def exit(self):
        # Triggers an exit event which should be detected in the process loop
        self.exit_event.set()

    def is_exiting(self):
        return self.exit_event.is_set()

from .abstract import FrameGrabberAbstract


def get_grabber(source, file_device=None, init_unpickable=True, **kwargs):
    """Return an image grabber for a source (e.g., video file, camera, another process, etc).
    The grabber is an object of a child of FrameGrabberAbstract class defined in abstract.py.
    A few provided standard methods are for grabbing frames from files and cameras. """

    grabber = None
    if isinstance(source, str):
        # Assume a predefined method
        if source == 'file':
            from .file_grabber import get_file_grabber
            grabber = get_file_grabber(filename=file_device, init_unpickable=init_unpickable, **kwargs)
        elif source == 'camera':
            from .opencv_grabber import FrameGrabberCV2Camera
            grabber = FrameGrabberCV2Camera(file_device, init_unpickable=init_unpickable, **kwargs)
            # print('Opened camera {} for frame grabbing.'.format(file_device))
    else:
        grabber = source
    assert isinstance(grabber, FrameGrabberAbstract)
    return grabber

def get_grabber(source, filename=None, init_unpickable=True, **kwargs):
    # Return an image grabber for a source (e.g., video file, camera, another process, etc).
    # The grabber is an object of a child of FrameGrabberAbstract class defined in abstract.py

    grabber = None
    if isinstance(source, str):
        # Assume a predefined method
        if source == 'file':
            from .file_grabber import get_file_grabber
            grabber = get_file_grabber(filename=filename, init_unpickable=init_unpickable, **kwargs)
    else:
        grabber = source
    return grabber

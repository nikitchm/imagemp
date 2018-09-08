def get_grabber(source, filename=None, init_unpickable=True, **kwargs):
    grabber = None
    if isinstance(source, str):
        # Assume a predefined method.
        if source == 'file':
            import os
            if not os.path.exists(filename):
                print("File doesn't exist: {}".format(filename))
                return None
            try:
                from opencv_grabber import FrameGrabberCV2File
                # First open the grabber fully (including the unpickable part, s.a. cv2.VideoCapture() in opencv),
                # to ensure that everything is working
                grabber = FrameGrabberCV2File(filename, init_unpickable=True, **kwargs)
                if not grabber.is_opened:
                    raise NameError('FailedOpenFile')
                if not init_unpickable:
                    # If everything is working, open the grabber without the unpickable part
                    grabber.close()
                    grabber = FrameGrabberCV2File(filename, init_unpickable=False, **kwargs)
                    print('Successfully opened {} using opencv'.format(filename))
            except NameError as e:
                # TODO : this part isn't brought up to pace
                try:
                    from skvideo_grabber import FrameGrabberSkvideoFile
                    grabber = FrameGrabberSkvideoFile(filename, **kwargs)
                    grabber.init_unpickable()
                    if not grabber.is_opened:
                        raise NameError('FailedOpenFile')
                except NameError as e:
                    print("Couldn't open video file")
    else:
        grabber = source
    return grabber
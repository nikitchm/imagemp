# if __name__ == '__main__' and __package__ == None:
if __name__ == '__main__':
    #     __package__ = "imagemp"
    import ctypes
    import argparse, os
    import imagemp as imp
    try:
        from imagemp.process_runners.examples.tracker_fish4 import FishTracker4
    except Exception as e:
        print(e)
    import time

    # Get input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="path to input video file")
    ap.add_argument("-r", "--recvid", required=False, help="path to recorded video file")
    args = vars(ap.parse_args())
    vid_filename = args['video'] if args['video'] is not None else \
        r'C:\data\rnd\Max_Experiments\behave\2017_10_31_ReaCh_pref\oxy-control_preference_1.vid.avi'
        # r'C:\data\rnd\Max_Experiments\behave\MTZCont_Optovin_fish5.vid.avi'
        # r'E:\10\22\optovinE3cond_fish50.vid.avi'
        # r'D:\rnd\LargeData\behavior_sample_2015_09_24\MTZcontrol_fish5_DMSO_0.3.vid.avi'
        # r'C:\data\rnd\Max_Experiments\behave\MTZcont_Optovin_fish5.vid.stimLED.avi'
        # r'/mnt/data/rnd/LargeData/MTZCont_Optovin_fish5.vid.avi'
    vid_av_filename = os.path.splitext(vid_filename)[0] + '_average.npy'

    # Get image shape:
    grabber = imp.get_grabber(source='file', filename=vid_filename, init_unpickable=True)
    if grabber is None or not grabber.is_opened:
        exit(1)
    im_shape = grabber.capture()[0][1].shape
    print(im_shape)
    grabber.close()
    # im_shape = (480, 480, 3)

    # Define data types
    timestamp_type = ctypes.c_float
    array_type = ctypes.c_uint16

    # Initialize the shared image data structure
    data_structure_type = 1
    if data_structure_type == 0:
        print('Using SharedSingleFrame')
        shared_d = imp.SharedSingleFrame(im_shape=im_shape,
                                         array_type=array_type,
                                         timestamp_type=timestamp_type,
                                         lock=False)
    elif data_structure_type == 1:
        print('Using SharedFrameList')
        shared_d = imp.SharedFrameList(im_shape=im_shape,
                                       nelem=100,
                                       array_type=array_type,
                                       timestamp_type=timestamp_type,
                                       lock=False)
    else:
        raise Exception("data_structure_type {} isn't defined".format(data_structure_type))

    # Create shared_events (can also be assigned from FrameGrabber after its initialization)
    shared_events = imp.SharedEvents()

    # Framegrabber scheduler
    fg_scheduler = imp.Scheduler(dt=0.005)
    grabber = imp.get_grabber(source='file', filename=vid_filename, init_unpickable=False)

    # Start the framegrabber
    frame_grabber = imp.FrameGrabberRunner(shared_data=shared_d,
                                           grabber=grabber,
                                           shared_events=shared_events,
                                           scheduler=fg_scheduler)

    # Start a display
    display = imp.SimpleDisplay(shared_data=shared_d,
                                shared_events=shared_events)

    # Start the recorder
    vfpath_split = os.path.splitext(vid_filename)
    vid_rec_filename = vfpath_split[0] + '_rec' + vfpath_split[1]
    recorder = imp.Recorder(shared_data=shared_d,
                            shared_events=shared_events,
                            filename=vid_rec_filename,
                            fourcc='XVID',
                            is_color=True,
                            im_shape=im_shape,
                            fps=30)

    # Start the analysis
    # analysis = FishTracker4(shared_data=shared_d,
    #                         shared_events=shared_events,
    #                         vid_av_filename=vid_av_filename)
    analysis = imp.Consumer()   # emtpy consumer process, complying with generic interprocess commands

    # Start the processes
    print('Starting the processes')
    display.start()
    recorder.start()
    # analysis.start()
    # time.sleep(1)
    frame_grabber.start()

    #
    for obj in [frame_grabber, display, recorder, analysis]:
        shared_events.add_mp_object(obj)

    t2run = 5
    time.sleep(t2run)
    print('{} sec have expired. Exiting.'.format(t2run))
    shared_events.exitall()

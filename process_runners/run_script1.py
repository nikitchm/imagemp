# if __name__ == '__main__' and __package__ == None:
if __name__ == '__main__':
    #     __package__ = "imagemp"
    import ctypes
    import argparse, os
    import imagemp
    from imagemp.process_runners.shared_events import SharedEvents
    from imagemp.scheduler.scheduler import Scheduler
    # from shframe_grabbers.factory import get_grabber
    from imagemp.shframe_grabbers.factory import get_grabber
    from imagemp.process_runners.frame_grabber import FrameGrabberRunner
    from imagemp.shared_frames.single_frame import SharedSingleFrame
    from imagemp.shared_frames.list_frames import SharedFrameList
    from imagemp.process_runners.examples.simple_display import SimpleDisplay
    from imagemp.process_runners.examples.recorder import Recorder
    from imagemp.process_runners.examples.tracker_fish4 import FishTracker4
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="path to input video file")
    ap.add_argument("-r", "--recvid", required=False, help="path to recorded video file")
    args = vars(ap.parse_args())
    vid_filename = args['video'] if args['video'] is not None else \
        r'E:\10\22\optovinE3cond_fish50.vid.avi'
        # r'C:\data\rnd\Max_Experiments\behave\MTZCont_Optovin_fish5.vid.avi'
        # r'D:\rnd\LargeData\behavior_sample_2015_09_24\MTZcontrol_fish5_DMSO_0.3.vid.avi'
        # r'C:\data\rnd\Max_Experiments\behave\MTZcont_Optovin_fish5.vid.stimLED.avi'
        # r'/mnt/data/rnd/LargeData/MTZCont_Optovin_fish5.vid.avi'
    vid_av_filename = os.path.splitext(vid_filename)[0] + '_average.npy'

    # Get image shape:
    grabber = get_grabber(source='file', filename=vid_filename, init_unpickable=True)
    if not grabber.is_opened:
        exit(1)
    im_shape = grabber.capture()[0][1].shape
    print(im_shape)
    grabber.close()
    # im_shape = (480, 480, 3)

    timestamp_type = ctypes.c_float
    # Initialize the shared data structure
    if 0:
        print('Using SharedSingleFrame')
        shared_d = SharedSingleFrame(im_shape=im_shape,
                                     array_type=ctypes.c_uint16,
                                     timestamp_type=timestamp_type,
                                     lock=False)
    else:
        print('Using SharedFrameList')
        shared_d = SharedFrameList(im_shape=im_shape,
                                   nelem=100,
                                   array_type=ctypes.c_uint16,
                                   timestamp_type=timestamp_type,
                                   lock=False)
    # Create shared_events (can also be taken from FrameGrabber after its initialization)
    shared_events = SharedEvents()

    # framegrabber scheduler
    fg_scheduler = Scheduler(dt=0.005)
    grabber = get_grabber(source='file', filename=vid_filename, init_unpickable=False)
    # Start the framegrabber
    frame_grabber = FrameGrabberRunner(shared_data=shared_d,
                                       grabber=grabber,
                                       shared_events=shared_events,
                                       scheduler=fg_scheduler)

    # Start a display
    display = SimpleDisplay(shared_data=shared_d,
                            shared_events=shared_events)

    # Start the recorder
    vfsplit = os.path.splitext(vid_filename)
    vid_rec_filename = vfsplit[0] + '_rec' + vfsplit[1]
    recorder = Recorder(shared_data=shared_d,
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

    display.start()
    # recorder.start()
    # analysis.start()
    time.sleep(1)
    frame_grabber.start()

    for obj in [frame_grabber, display, recorder, analysis]:
        shared_events.add_mp_object(obj)
    # for obj in [frame_grabber, display, recorder]:
    #     shared_events.add_mp_object(obj)

    t2run = 5
    time.sleep(t2run)
    print('{} sec have expired. Exiting.'.format(t2run))
    shared_events.exitall()

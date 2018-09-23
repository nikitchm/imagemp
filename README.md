# Imagemp Documentation

This package implements producer-consumer**s** paradigm for parallel acquisition and processing of images (or array in general), with a particular focus on speed.

The main approach is to start every task in a separate process using the multiprocessing library. The cornerstone is the acquisition process which should be able to acquire, store and make available to other processes the images at a required or maximum achievable pace.

In the simplest method, an image is grabbed from a source, s.a. a camera or a file, into a memory block directly shared with the consumer processes. This saves a significant amount of time (on the order of 1-3 milliseconds) required to copy data between the processes. However, there is still a risk of the consumer processes missing the read frames in cases where the consumer computations take longer than the frame acquisition time.

In order to alleviate the problem with missing frames, two shared frames structures are implemented. The first is the circular buffer of shared elements of a predefined length, which allows for some (ideally predictable) lag of the consumers relative to the producer. The second is a combination of the same circular buffer and a list (of an arbitrary length, but not shared between the processes) -- stored in the same process as the producer, or on disk via memmapping, and transfered into the circular buffer as it's freeing up.

The final structure requires the user to define the type and parameters of the shared-frame object, the time table for the frame acquisitions (either via the scheduler, or by manual triggering), as well as the consumer process objects. Afterwards, the producer and consumers are run in their own loops and processes, exchanging messages via the predefined shared structures and events.

Exra care is taken not to use CPU time for idle looping -- in all the cases, the processes, should they finish the current computations earlier than the new data becomes available, wait for semaphore events to continue running.

All the classes are designed in a hierarchical fashion: an abstract class defines a minimal set of methods expected from the corresponding class of classes (however, abstract classes from the abc library are not used here by design -- to simplify the understanding and usage for the beginners); If possible, the abstract class is inherited by a more specific child/parent class which implements as much as possible of the assumed shared functionality between the assumed children. Ultimately, most methods are broken down in sufficiently small pieces to allow for simple recycling by the end user by method overloading.   

As a good starting point, go through the process_runners/run_script1.py file. Also, have a look into the docs/ folder for more information.

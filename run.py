"""
Code by Muhammad Nouman Ahsan(Original Author)

Differnt Faces Recognition in Videos
IDE: VS Code(you can use it in other IDEs)

"""


# import necessary packages and methods
import os
import argparse
from utils.utils import _create_structure
from utils.inference import inference_recorder, inference_video
import datetime

# list of video extensions that the program will check
# add other video extensions if not in the list
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv']

# a function which iterates through the input path, pick videos one by one and run inference on it
# it saves the results in the respective folders in the output results folder
def run(args):
    if args.mode == "video":
        # pick files in the input path directory, and iterate one by one
        for file in os.listdir(args.input_path):

            # extract the extension and the name of the file
            basename, extension = os.path.splitext(file)

            # check if the file extension is in the VIDEO_EXTENSIONS list. I means the file is video if it exists
            # if it's not a video then proceed to the next file in the folder
            if extension not in VIDEO_EXTENSIONS:
                print("file found was not a video. Proceeding to next file...")
                continue
            
            # _create_structure creates folders in the output path based on video name in which results are to be saved
            _create_structure(args.output_path, basename)
            video_path = os.path.join(args.input_path, file)

            # note the current time and call inference function imported from utils folder
            # pass all the necessary arguments in the function
            start_time = datetime.datetime.now()
            inference_video(
                video_path,
                os.path.join(args.output_path, basename),
                distance_threshold=args.distance_threshold,
                resize_scale=args.resize_scale, 
                gpu=args.gpu,
                step=args.step,
                frames_difference_threshold=args.frames_difference,
                preview=args.view,
                verbose=args.verbose
                )

            # note the end time and calculate the total time taken for processing the video
            end_time = datetime.datetime.now()
            _process_time_taken = end_time - start_time
            print(f"time taken to process video: {file} is {_process_time_taken}...")
            
        print()
        print(f"Please check {args.output_path} directory for results...")

    # if arg is cam then run the inference on webcam
    elif args.mode == "recorder":

        # _create_structure creates folders in the output path based on video name in which results are to be saved
        _create_structure(args.output_path, args.keyword)

        # note the current time and call inference function imported from utils folder
        # pass all the necessary arguments in the function
        start_time = datetime.datetime.now()
        inference_recorder(
            args.keyword,
            os.path.join(args.output_path, args.keyword),
            distance_threshold=args.distance_threshold,
            resize_scale=args.resize_scale, 
            gpu=args.gpu,
            null_frames_threshold=args.null_frames_threshold,
            frames_difference_threshold=args.frames_difference,
            preview=args.view,
            verbose=args.verbose
            )

        # note the end time and calculate the total time taken for processing the video
        end_time = datetime.datetime.now()
        _process_time_taken = end_time - start_time
        print(f"time taken to process video is {_process_time_taken}...")
        
        print()
        print(f"Please check {args.output_path} directory for results...")
    
    # else raise a value error
    else:
        raise ValueError('mode must be either video or recorder')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Program input flags
    # These flags can be set in the command line interface when running the command
    # sample command is 
    # $ python run.py --flag1=value --flag2=value ...
    parser.add_argument('--mode', type=str, required=False, default="video", 
        help='Mode to run the program accordingly. Default values; video, recorder, default=%(default)s')

    parser.add_argument('--input_path', type=str, required=False, default="videos", 
        help='Path to the input directory where videos are placed, default=%(default)s')

    parser.add_argument('--output_path', type=str, required=False, default="results", 
        help='path to output directory where results will be saved, default=%(default)s')

    parser.add_argument('--keyword', type=str, required=False, default="Kaltura",
        help='keyword is any keyword in windows title, only working with recorder mode, default=%(default)s')

    parser.add_argument('--null_frames_threshold', type=str, required=False, default=15,
        help='stop the processing if no face is detected this times, only work with recorder, default=%(default)s')
        
    parser.add_argument('--view', default=False, action="store_true",
        help='Preview Video and plots during processing, default it False')

    parser.add_argument('--step', default=60, type=int,
        help='pick nth frame to process in the vidoe. default is 30 meaning take a frame after 1 sec if video is 30FPS')

    parser.add_argument('--frames_difference', type=int, default=5,
        help='A new face is recognised in nth consecutive frames will be considered as a new person, default is 10')

    parser.add_argument('--gpu', default=False, action="store_true",
        help = "Specify whether to use GPU, default=%(default)s")

    parser.add_argument('--distance_threshold', type=float, required=False, default=0.5,
        help = "How much distance between faces to consider it a match. Lower is more strict. 0.5 is typical best performance., ")

    parser.add_argument('--resize_scale', type=float, required=False, default=0.5,
        help = "Resize the input video, e.g. 0.75 will resize to 1/4th of the video. default is 1 means No resize")

    parser.add_argument('--verbose', type=int, required=False, default=1,
        help = "shows the log level in the cosole. Set it to zero to show nothing in the console") 
    
    # parsing the flags into a variable called args
    args = parser.parse_args()

    # check if the input videos path exists
    if not os.path.exists(args.input_path):
        raise ValueError('input path does not exists')

    # check if the output results path exists
    if not os.path.exists(args.output_path):
        raise ValueError('output path does not exists')
    
    # if verbose value is other than 0 and 1, then set it to 1
    args.verbose = int(args.verbose) if int(args.verbose) in [0, 1] else 1

    # pass the arguments to the run function and call it
    run(args)
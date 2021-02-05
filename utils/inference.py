"""
Code by Muhammad Nouman Ahsan(Original Author)

Differnt Faces Recognition in Videos
IDE: VS Code(you can use it in other IDEs)

"""


import os
import cv2
import numpy as np
from tqdm import tqdm
import face_recognition
import matplotlib.pyplot as plt
from utils.utils import crop_face, _take_average
import datetime
import win32gui
import win32ui
import win32con
import ctypes
from ctypes import windll
from PIL import Image

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)

global WINDOW
WINDOW = None


def _add_to_references(references, _encodings, frame_ori, face_final, output_directory, people_count):
    # crop face from the frame based on bounding boxes detected; crop_face function is called here
    _face_image = crop_face(frame_ori, face_final[0])
    _name = "Person_" + str(people_count)
    images_path = os.path.join(output_directory, "images", _name + ".jpg")
    # resizing the croped face to specific size
    _face_image = cv2.resize(_face_image, (224, 192), interpolation=cv2.INTER_AREA)
    # write text on the image
    cv2.putText(_face_image, _name, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)
    # finally save the image
    cv2.imwrite(images_path, _face_image)

    # then add the information to references list
    references.append({
        "encodings": _encodings,
        "name": _name,
        "count": 0,
        "frames_appeared": 0,
        "duration": "00:00:00"
    })
    return references


def plot_summary(references, output_directory, preview):
    # this plot contains sub plots based on number of people found in the video; subplots are images of people
    images_path = os.path.join(output_directory, "images")
    size = int(np.ceil(len(references) / 5))

    if len(references) == 0:
        return

    if len(references) <= 5:
        fig, ax = plt.subplots(nrows=size, ncols=len(references))
        for col in range(len(references)):
            if col >= len(references):
                break
            title = f"duration\n{references[col]['duration']}"

            # read the image saved in the output direcotry based on their names
            image = cv2.imread(os.path.join(images_path, references[col]["name"] + ".jpg"))

            # convert image from BGR to RGB image
            image = image[:, :, ::-1]
            if len(references) == 1:
                # plot the image
                ax.imshow(image)
                ax.set_xlabel(title, fontsize=8)
                # Turn off tick labels
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            else:
                ax[col].imshow(image)
                ax[col].set_xlabel(title, fontsize=8)
                # Turn off tick labels
                ax[col].set_yticklabels([])
                ax[col].set_xticklabels([])
    else:
        fig, ax = plt.subplots(nrows=size, ncols=5)
        for row in range(size):
            for col in range(5):
                index = (row * 5) + col
                if index >= len(references):
                    break

                title = f"duration: {references[index]['duration']}"
                image = cv2.imread(os.path.join(images_path, references[index]["name"] + ".jpg"))
                image = image[:, :, ::-1]
                ax[row][col].imshow(image)
                ax[row][col].set_xlabel(title, fontsize=8)
                # Turn off tick labels
                ax[row][col].set_yticklabels([])
                ax[row][col].set_xticklabels([])

    # save the plot
    fig.savefig(os.path.join(output_directory, 'plot.jpg'), format='jpg', dpi=800)
    if preview:
        plt.show()


def plot_hist(references, output_directory, preview):

    if len(references) == 0:
        return

    # make lists of people found, their number of frames appeared and duration
    _people = [reference["name"].split("_")[-1] for reference in references]
    _frames_appeared = [reference["frames_appeared"] for reference in references]
    _duration = [reference["duration"] for reference in references]

    def _label_plot(bar_plot, _duration):
        for rect, _dur in zip(bar_plot, _duration):
            height = rect.get_height()
            text = f"duration: {_dur}"
            plt.text(rect.get_x() + rect.get_width()/2.0, height, text, ha='center', va='bottom')

    fig = plt.figure(figsize=(20, 20))
    # creating the bar plot 
    plot = plt.bar(_people, _frames_appeared, color='blue', width=0.4)

    # label the bars with their duration values
    _label_plot(plot, _duration)
    plt.xlabel("People Appeared in Video") 
    plt.ylabel("No. of frames appeared in") 
    plt.title("People found in video frames")

    # save the plot to output folder
    fig.savefig(os.path.join(output_directory, 'hist.jpg'), format='jpg', dpi=200)

    # preview is True then show the plot, 
    if preview:
        plt.show()


def inference_video(
    video_path,
    output_directory,
    distance_threshold=0.7,
    resize_scale=0.75,
    gpu=False,
    step=30,
    frames_difference_threshold=10,
    preview=False,
    verbose=1):

    # initialize all the necessary variables
    current_frame_index = 0     # holds current frame index
    previous_frame_index = 0    # holds previous frame index
    people_count = 1    # number of people found in the video
    temp_encodings = []     # list holding face data
    references = []     # list which keep information about all people found in the video
    _temp_referece = None   # temp variable
    is_different_count = 0  # holds faces differnce count
    _step = step    # hold number of frames to skip

    # read video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # find total video frames
    FPS = cap.get(cv2.CAP_PROP_FPS)

    # if FPS value is None then set it to 30 by default
    FPS = FPS if FPS is not None and FPS > 0 else 30

    # compute approx time to process the video
    approx_processing_time = total_frames /(FPS)
    print(f"Approx processing time for the video is: {str(datetime.timedelta(seconds=approx_processing_time))}")
    print("Process started...")

    # function to bring time to proper format
    def _round_seconds(_time):
        h, m, s = [_time.split(':')[0],
                    _time.split(':')[1],
                    str(round(float(_time.split(':')[-1])))]
        return h + ':' + m + ':' + s

    # start inference
    while(True):
        # read frame based on current frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame_ori = cap.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame_ori[:, :, ::-1]

        # break the loop if current frame index exceeds the total frames (no more frames to process)
        if current_frame_index >= total_frames - 1:
            break

        if ret:
            # detect faces in the frame
            # processing with gpu is faster. Specify gpu flag in cmd if gpu is available
            if gpu:
                face_bboxes = face_recognition.face_locations(frame, model="cnn")  
            else: 
                face_bboxes = face_recognition.face_locations(frame)
            
            # ignore the frame with no faces detected or multiple faces detected
            if len(face_bboxes) > 0:
                # pick up only one face (having largest area) if multiple faces are detected
                # face_area is area of a face with largest area
                face_area = max([(face[2] * face[3]) for face in face_bboxes])
                face_final = [face for face in face_bboxes if (face[2] * face[3]) == face_area]
                
                # find face encodings (list of 128 values) which will be used for face recognition
                if gpu:
                    face_enc = face_recognition.face_encodings(frame, known_face_locations=face_final, model="cnn")[0]
                else: 
                    face_enc = face_recognition.face_encodings(frame, known_face_locations=face_final)[0]

                matches = []
                # references contains people found in faces (initially it is empty)
                if len(references) > 0:
                    # picking up face data for all people found
                    known_face_encodings = [item["encodings"] for item in references]

                    # comparing all known faces to the one detected in the frame. A list of matches containing true/false is returned
                    matches = face_recognition.compare_faces(known_face_encodings, face_enc, tolerance=distance_threshold)

                if any(matches):
                    _step = step
                    # if there's a match then find out the matching person from the references 
                    best_match_index = matches.index(True)
                    # add count to the matched person
                    references[best_match_index]["count"] += 1
                    # increament number of frames appeared in for that person
                    references[best_match_index]["frames_appeared"] += _step
                    # compute duration for that person based on frames appeared and video FPS
                    # e.g. 60 frames and 30 FPS means 60/30 equals 2 seconds
                    _frames_appeared = references[best_match_index]["frames_appeared"]
                    _duration_sec = _frames_appeared / FPS
                    _duration_standard = str(datetime.timedelta(seconds=_duration_sec))
                    references[best_match_index]["duration"] = _round_seconds(_duration_standard)
                    # print information
                    if verbose == 1:
                        print(f'match found for: {references[best_match_index]["name"]} | count: {references[best_match_index]["count"]} | duration: {_duration_standard}')

                    # reset values
                    is_different_count = 0
                    temp_encodings = []

                # if match was not found, meaning the person found was a different person
                else:
                    if verbose == 1:
                        print("New face found | times: ", is_different_count)

                    # _temp_referece contains face data for new face. We want to detect a new face in frames_difference_threshold number
                    # of consecutive frames before considering it a new face.
                    if _temp_referece is not None:
                        # maching the current new face with the new face found in the previous frame (ensure that the new face found
                        # in consecutive face is the same) 
                        match = face_recognition.compare_faces([_temp_referece], face_enc, tolerance=distance_threshold)
                        if not match[0]:
                            temp_encodings = []
                            is_different_count = 0

                    # increament is_different_count and lower the step value
                    is_different_count += 1
                    _step = 2
                    # add new face data to temp list
                    temp_encodings.append(face_enc)
                    # make the face data as a reference
                    _temp_referece = face_enc
                    # consider the new face found frames_difference_threshold times and add it to references list
                    if is_different_count == frames_difference_threshold:
                        # take average of new face data stored in temo_encodings (for better results)
                        _encodings = _take_average(temp_encodings)
                        # call function to add to references
                        references = _add_to_references(references, _encodings, frame_ori, face_final, output_directory, people_count)
                        print(str(people_count) + " new faces detected...")
                        people_count += 1
                        temp_encodings = []
                        _step = step
                        is_different_count = 0

            # if preview is True then show the video
            if preview:
                cv2.imshow("window", frame_ori)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    current_frame_index = min(total_frames - 1, current_frame_index + _step)

        # set current_frame_index
        current_frame_index = min(total_frames - 1, current_frame_index + _step)
    
    print()
    print("Video processed. Now making plots...")
    # plot results stored in references
    plot_summary(references, output_directory, preview)
    plot_hist(references, output_directory, preview)

    # Release all space and windows once done 
    cap.release()
    cv2.destroyAllWindows()


def winEnumHandler(hwnd, keyword):
    if win32gui.IsWindowVisible(hwnd):
        win = win32gui.GetWindowText(hwnd)
        if keyword in win:
            global WINDOW
            WINDOW = win


def inference_recorder(
    window_keyword,
    output_directory,
    distance_threshold=0.7,
    resize_scale=0.5,
    gpu=False,
    frames_difference_threshold=10,
    null_frames_threshold=30,
    preview=False,
    verbose=1):

    win32gui.EnumWindows(winEnumHandler, window_keyword)
    hwnd = win32gui.FindWindow(None, WINDOW)
    print(WINDOW)

    # Change the line below depending on whether you want the whole window or just the client area
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    window_width = right - left
    window_height = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, window_width, window_height)
    saveDC.SelectObject(saveBitMap)

    # initialize all the necessary variables
    frames_processed = 0     # number of frames processed
    null_frames = 0     # frames processed without any face detected
    people_count = 1    # number of people found in the video
    temp_encodings = []     # list holding face data
    references = []     # list which keep information about all people found in the video
    _temp_referece = None   # temp variable
    is_different_count = 0  # holds faces differnce count

    FPS = 25
    FPS_ACTUAL = 25
    print("Process started...")

    # function to bring time to proper format
    def _round_seconds(_time):
        h, m, s = [_time.split(':')[0],
                    _time.split(':')[1],
                    str(round(float(_time.split(':')[-1])))]
        return h + ':' + m + ':' + s
    
    start_time = datetime.datetime.now()
    # start inference
    while(True):
        # get the frame from the recorder
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        # convery the image into numpy array (opencv format)
        frame_ori = np.array(img)
        frame = cv2.resize(frame_ori, (0, 0), fx=resize_scale, fy=resize_scale)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]
        # break the loop if current frame index exceeds the total frames (no more frames to process)
        if null_frames >= null_frames_threshold:
            break

        # detect faces in the frame
        # processing with gpu is faster. Specify gpu flag in cmd if gpu is available
        if gpu:
            face_bboxes = face_recognition.face_locations(frame, model="cnn")  
        else: 
            face_bboxes = face_recognition.face_locations(frame)
        
        # ignore the frame with no faces detected or multiple faces detected
        if len(face_bboxes) > 0:
            null_frames = 0
            # pick up only one face (having largest area) if multiple faces are detected
            # face_area is area of a face with largest area
            face_area = max([(face[2] * face[3]) for face in face_bboxes])
            face_final = [face for face in face_bboxes if (face[2] * face[3]) == face_area]
            
            # find face encodings (list of 128 values) which will be used for face recognition
            if gpu:
                face_enc = face_recognition.face_encodings(frame, known_face_locations=face_final, model="cnn")[0]
            else: 
                face_enc = face_recognition.face_encodings(frame, known_face_locations=face_final)[0]

            matches = []
            # references contains people found in faces (initially it is empty)
            if len(references) > 0:
                # picking up face data for all people found
                known_face_encodings = [item["encodings"] for item in references]

                # comparing all known faces to the one detected in the frame. A list of matches containing true/false is returned
                matches = face_recognition.compare_faces(known_face_encodings, face_enc, tolerance=distance_threshold)

            if any(matches):
                # if there's a match then find out the matching person from the references 
                best_match_index = matches.index(True)

                # calculate frames appeared
                frames_skipped = FPS / FPS_ACTUAL
                references[best_match_index]["frames_appeared"] += frames_skipped
                # compute duration for that person based on frames appeared and video FPS
                # e.g. 60 frames and 30 FPS means 60/30 equals 2 seconds
                _frames_appeared = references[best_match_index]["frames_appeared"]
                _duration_sec = _frames_appeared / FPS
                _duration_standard = str(datetime.timedelta(seconds=_duration_sec))
                references[best_match_index]["duration"] = _round_seconds(_duration_standard)
                # print information
                if verbose == 1:
                    print(f'match found for: {references[best_match_index]["name"]} | duration: {_duration_standard}')

                # reset values
                is_different_count = 0
                temp_encodings = []

            # if match was not found, meaning the person found was a different person
            else:
                if verbose == 1:
                    print("New face found | times: ", is_different_count)

                # _temp_referece contains face data for new face. We want to detect a new face in frames_difference_threshold number
                # of consecutive frames before considering it a new face.
                if _temp_referece is not None:
                    # maching the current new face with the new face found in the previous frame (ensure that the new face found
                    # in consecutive face is the same) 
                    match = face_recognition.compare_faces([_temp_referece], face_enc, tolerance=distance_threshold)
                    if not match[0]:
                        temp_encodings = []
                        is_different_count = 0

                # increament is_different_count and lower the step value
                is_different_count += 1
                # add new face data to temp list
                temp_encodings.append(face_enc)
                # make the face data as a reference
                _temp_referece = face_enc
                # consider the new face found frames_difference_threshold times and add it to references list
                if is_different_count == frames_difference_threshold:
                    # take average of new face data stored in temo_encodings (for better results)
                    _encodings = _take_average(temp_encodings)
                    # call function to add to references
                    references = _add_to_references(references, _encodings, frame, face_final, output_directory, people_count)
                    print(str(people_count) + " new faces detected...")
                    people_count += 1
                    temp_encodings = []
                    is_different_count = 0

        # if no face was detected in the video
        else:
            null_frames += 1
            print(f"no face detected for {null_frames} consecutive frames. Stopping threshold: {null_frames_threshold}")

        frames_processed += 1
        # time calculation
        end_time = datetime.datetime.now()
        time_diff = end_time - start_time
        FPS_ACTUAL = frames_processed / time_diff.seconds if time_diff.seconds != 0 else 25

    print()
    print("Video processed. Now making plots...")
    # plot results stored in references
    plot_summary(references, output_directory, preview)
    plot_hist(references, output_directory, preview)

    # Release all space and windows once done 
    cv2.destroyAllWindows()
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
#################################################################
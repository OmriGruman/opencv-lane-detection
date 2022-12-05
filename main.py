import cv2
import matplotlib.pyplot as plt
import numpy as np


def print_img(image,name="image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(name)
    plt.show()


def read_frames_from_video(video_path, num_seconds=30):
    frames = list()
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_seconds * fps > total_frames:
        print(f'Cannot read {num_seconds} seconds from video...')

    num_frames = min(num_seconds * fps, total_frames)
    num_seconds = num_frames / fps

    print(f"Start reading {num_frames} frames ({num_seconds:.2f} seconds) from {video_path} ({fps} fps)")

    for i in range(num_frames):
        image = capture.read()[1]
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        frames.append(image)
    capture.release()

    print(f"Successfully read {len(frames)} frames from {video_path}")

    return frames, fps

last_left = None
last_right = None

def preprocess_frames(raw_frames):
    global last_left
    global last_right

    res_frames = raw_frames
    # Ideas:
        # Crop image
        # Filter only certain color pixels
        # Dilate over lanes to connect broken lines into one lane (- - - -) -> (-------)
        # How to deal with shades (going under bridge)?
            # Normalize colors? 
            # Intensify dark color?
            # Contrast?
    h, w, c = res_frames.shape
    top = h*65//100

    # crop 
    cropped = raw_frames[top:, :, :]
    #print_img(cropped, 'crop')

    # detect white lines
    filtered_frame = cv2.inRange(cropped, np.array([180, 180, 180]), np.array([250, 250, 250]))
    #print_img(filtered_frame, 'filter')

    # avoid irrelevant lanes
    e_kernel = np.ones((3,1))
    eroded_frame = cv2.erode(filtered_frame, e_kernel)
    #print_img(eroded_frame, 'erode')

    # stay with minimal lines
    for row in range(eroded_frame.shape[0]):
        ids = [[]]
        for col in range(eroded_frame.shape[1]):
            if eroded_frame[row, col] > 0:
                if len(ids[-1]) > 0:
                    if col - ids[-1][-1] > 10:
                        ids.append([])
                ids[-1].append(col)
        
        if len(ids[-1]) > 0:
            for ii in ids:
                mid = (ii[0] + ii[-1]) // 2
                eroded_frame[row, ii] = 0
                eroded_frame[row, mid:mid+1] = 255
    #print_img(eroded_frame, 'shrink')

    for th in range(10, 3, -1):
        lines = cv2.HoughLines(eroded_frame, 1, np.pi / 180, th)

        if lines is not None:
            lines = sorted(lines[:, 0, :], key=lambda line: line[1])
            lines = list(filter(lambda line: 35 > abs(line[1]*180/np.pi - 90) > 5, lines))   

            # count lines for each side
            left_lines = list(filter(lambda line: line[1] < (np.pi / 2), lines))
            num_left = len(left_lines)
            right_lines = list(filter(lambda line: line[1] > (np.pi / 2), lines))
            num_right = len(right_lines)
            
            # choose best lines
            if num_left > 5:
                
                if last_left is None:
                    left_line = left_lines[num_left//2]
                elif len(left_lines) == 0:
                    left_line = last_left
                else:
                    left_line = sorted(left_lines, key=lambda line: abs(line[1] - last_left[1]))[0]

            if num_right > 5:
                if last_right is None:
                    right_line = right_lines[num_left//2]
                elif len(right_lines) == 0:
                    right_line = last_right
                else:
                    right_line = sorted(right_lines, key=lambda line: abs(line[1] - last_right[1]))[0]

                last_left = left_line
                last_right = right_line
                break

    left_line = last_left
    right_line = last_right

    if lines is not None:
        all_lines = cropped.copy()
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            all_lines = cv2.line(all_lines, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        #print_img(all_lines, f'th:{th}, lines:{len(lines)}')

    for rho, theta in [left_line, right_line]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cropped = cv2.line(cropped, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
    #print_img(cropped, f'final')

    return res_frames


def detect_lanes(preprocessed_frames):
    res_frames = preprocessed_frames
    # How to find best lines?
        # Canny
        # Hough Parabula ?? self implement (Omri) or use HoughLines (Mark)
    # How to detect lane transition?
        # transform lines from (x,y) to (rho, theta) and detect changes in rho (distance to origin)
            # Distance is decreasing -> LEFT
            # Distance is increasing -> RIGHT 
            # Use threshold to deal with noise
    # How to follow correct lanes during transition?
        # Compare to last frame
        # Lanes change will happen naturally when transition is complete
    return res_frames


def save_frames_to_video(frames, fps):
    video_path = 'lanes.mp4'
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    color = True
    
    writer = cv2.VideoWriter(video_path, codec, fps, (w, h), color)

    print(f"Writing {len(frames)} frames to {video_path} ({fps} fps)")

    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == '__main__':

    input_video_path = 'video1.mp4'    
    raw_frames, fps = read_frames_from_video(input_video_path)

    preprocessed_frames = [preprocess_frames(frame) for frame in raw_frames]
    final_frames = detect_lanes(preprocessed_frames)

    save_frames_to_video(final_frames, fps)
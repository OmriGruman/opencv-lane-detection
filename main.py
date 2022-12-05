import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


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
    height, width, channel = res_frames.shape
    top = height*65//100

    # crop 
    cropped = res_frames[top:, :, :]
    #print_img(cropped, 'crop')

    # detect white lines
    filtered_frame = cv2.inRange(cropped, np.array([180, 180, 180]), np.array([250, 250, 250]))
    #print_img(filtered_frame, 'filter')

    # avoid irrelevant lanes
    e_kernel = np.ones((3,1))
    eroded_frame = cv2.erode(filtered_frame, e_kernel)
    #print_img(eroded_frame, 'erode')

    # stay with minimal lines
    squashed_frame = filtered_frame
    for row in range(squashed_frame.shape[0]):
        ids = [[]]
        for col in range(squashed_frame.shape[1]):
            if squashed_frame[row, col] > 0:
                if len(ids[-1]) > 0:
                    if col - ids[-1][-1] > 10:
                        ids.append([])
                ids[-1].append(col)
        
        if len(ids[-1]) > 0:
            for ii in ids:
                mid = (ii[0] + ii[-1]) // 2
                squashed_frame[row, ii] = 0
                squashed_frame[row, mid:mid+1] = 255
    #print_img(squashed_frame, 'squash')

    # RANSAC
    points = np.argwhere(squashed_frame)
    d_th = 2
    lines = []
    for i in range(100):
        p1, p2 = np.random.choice(points.shape[0], size=2, replace=False)
        (y1, x1), (y2, x2) = points[[p1, p2]]

        inliers = []
        for y0, x0 in points:
            d = np.linalg.norm(np.cross((x1 - x0, y1 - y0), (x2 - x1, y2 - y1))) / np.linalg.norm((x2 - x1, y2 - y1))
            if d <= d_th:
                inliers.append([x0, y0])

        if len(inliers) < 10: continue

        inliers = np.array(inliers)
        X = np.concatenate((inliers[:, 0].reshape(-1, 1), np.ones((inliers.shape[0], 1))), axis=1)
        Y = inliers[:, 1].reshape(-1, 1)        
        w, b = np.linalg.lstsq(X, Y, rcond=None)[0].flatten()

        if abs(w) < 0.3: continue
        if w > 0 and (-b/w) < 210: continue
        if w < 0 and (-b/w) > 210: continue

        # if different from all lines up to this moment
        if all([abs(w0 - w) > 0.1 or abs(b0 - b) > 10 for w0, b0, _ in lines]):
            lines.append([w, b, inliers.shape[0]])

    lines = sorted(lines, key=lambda line: -line[2])    
    #print('\n'.join(list(map(str, lines))))    

    right_lines = list(filter(lambda l: l[0] > 0, lines))
    left_lines = list(filter(lambda l: l[0] < 0, lines))

    if len(left_lines) > 0:
        left_lines = sorted(left_lines, key=lambda l: l[1]/l[0])
        last_left = left_lines[0]

    if len(right_lines) > 0:
        right_lines = sorted(right_lines, key=lambda l: -l[1]/l[0])
        last_right = right_lines[0]

    # draw lines
    for m, n, num_inliers in [last_left, last_right]:
        x1 = 0
        y1 = round(m * x1 + n)
        x2 = cropped.shape[1] - 1
        y2 = round(m * x2 + n)
        cropped = cv2.line(cropped, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        #print(f'line: y = {m:.2f}x + {n:.2f}, inliers{num_inliers}, lines:{len(lines)}')
        #print_img(all_lines, f'line: y = {m:.2f}x + {n:.2f}, inliers{num_inliers}, lines:{len(lines)}')
    #print_img(res_frames, f'lines: {len(lines)}')
    '''
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
    '''

    res_frames[top:, :, :] = cropped
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
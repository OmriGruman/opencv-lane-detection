import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from time import time 


def print_img(image,name="image"):
    return
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(name)
    plt.show()


def read_frames_from_video(video_path):
    start = 18
    finish = 270

    frames = list()
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    

    start_frame = int(start * fps)
    finish_frame = int(finish * fps)
    num_frames =  finish_frame - start_frame
    num_seconds = num_frames / fps

    print(f"Start reading {num_frames} frames ({num_seconds:.2f} seconds) from {video_path} ({fps} fps)")

    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    for i in range(num_frames):
        image = capture.read()[1]
        frames.append(image)
    capture.release()

    print(f"Successfully read {len(frames)} frames from {video_path}")

    return frames, fps


def draw_line(img, line):
    m, n = line

    y1 = 0
    x1 = round(m * y1 + n)
    y2 = img.shape[0] - 1
    x2 = round(m * y2 + n)

    return cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
    

def draw_lane(img, left, right):
    lane = img.copy()
    lane[:,:,:] = 0

    point = lambda y, m, n: [m * y + n, y]

    ml, nl = left
    mr, nr = right
    y_top = 0
    y_bottom = img.shape[0] - 1

    top_left = point(y_top, ml, nl)
    bottom_left = point(y_bottom, ml, nl)
    top_right = point(y_top, mr, nr)
    bottom_right = point(y_bottom, mr, nr)

    points = np.round([[top_left, top_right, bottom_right, bottom_left]]).astype(int)
    cv2.fillPoly(lane, np.round([points]).astype(int), color=[255, 255, 255])
    
    lane_mask = lane > 0    
    img[:, :, 1][lane_mask[:, :, 1]] = 255

    return img


def squash_lanes(img, max_dist):
    for row in range(img.shape[0]):
        ids = [[]]
        for col in range(img.shape[1]):
            if img[row, col] > 0:
                if len(ids[-1]) > 0:
                    if col - ids[-1][-1] > max_dist:
                        ids.append([])
                ids[-1].append(col)
        
        if len(ids[-1]) > 0:
            for ii in ids:
                mid = (ii[0] + ii[-1]) // 2
                img[row, ii] = 0
                img[row, mid:mid+2] = 255
    return img


def poly_crop(img):
    poly_mask = np.zeros_like(img)
    mask_value = 255
    
    top_left = (310, 0)
    top_right = (500, 0)
    mid_right = (img.shape[1] - 1, 100)
    mid_left = (0, 100)
    bottom_right = (img.shape[1] - 1, img.shape[0] - 1)
    bottom_left = (0, img.shape[0] - 1)

    points = np.array([[
        top_left, 
        top_right,
        mid_right, 
        bottom_right, 
        bottom_left, 
        mid_left
    ]], dtype=np.int32)
    
    cv2.fillPoly(poly_mask, points, mask_value)

    return cv2.bitwise_and(img, poly_mask), poly_mask


def preprocess_frames(raw_frame, skip):
    res_frame = raw_frame

    height, _, _ = res_frame.shape
    top = height * 62 // 100
    
    # crop 
    cropped = res_frames[top:, :, :]

    # detect edges lines
    blur = cv2.GaussianBlur(cropped, (5,5), 0)
    canny = cv2.Canny(blur, 100, 200)

    # crop polygon
    polygon, poly_mask = poly_crop(canny)

    # stay with minimal lines
    squash = squash_lanes(img=polygon, max_dist=40)

    return cropped, top, squash    


prev_right = None
prev_left = None
slope_border = 0
slope_left = -3.5
slope_right = 3.5
middle = None
turn = 0

num_samples = 100
shoulder_threshold = 1.5
inliers_threshold = 30

def ransac_lines(points, shoulder_size, min_inliers, min_left_slope, max_right_slope, middle_slope, vanishing_point):

    for i in range(num_samples):
        p1, p2 = np.random.choice(points.shape[0], size=2, replace=False)

        (y1, x1), (y2, x2) = points[[p1, p2]]
        inliers = []
        inds = []

        for i, (y0, x0) in enumerate(points):
            d = np.linalg.norm(np.cross((x1 - x0, y1 - y0), (x2 - x1, y2 - y1))) / np.linalg.norm((x2 - x1, y2 - y1))
            if d <= shoulder_size:
                inliers.append([y0, x0])
                inds.append(i)

        # focus on lines that fit well in a line
        if len(inliers) < min_inliers: continue

        # fit line to all inliers, but according to inverse dimensions (y - horizontal axis, x - vertical axis)
        inliers = np.array(inliers)
        X = np.concatenate((inliers[:, 0].reshape(-1, 1), np.ones((inliers.shape[0], 1))), axis=1)
        Y = inliers[:, 1].reshape(-1, 1)        
        w, b = np.linalg.lstsq(X, Y, rcond=None)[0].flatten()
        
        # filter slopes in certain range
        if w < min_left_slope or w > max_right_slope: continue

        # filter lines according to intersect
        if vanishing_point:
            # right lines
            if w > 0 and b < (vanishing_point - 20): continue
            # left lines
            if w < 0 and b > (vanishing_point + 20): continue

        # right lane
        if w > middle_slope:
            if right_line is None or len(inliers) > right_line[2]:
                right_line = [w, b, len(inliers)]
        # left lane
        else:
            if left_line is None or len(inliers) > left_line[2]:
                left_found = True
                left_line = [w, b, len(inliers)]


def verify_lines(left_line, prev_left, right_line, prev_right):
    left = left_line
    right = right_line

    if prev_left:
        if (left_line == None or 
            (left_line[0] > prev_left[0] and left_line[1] < prev_left[1] - 10) or
            (left_line[0] < prev_left[0] and left_line[1] > prev_left[1] + 10) or 
            (not turn and abs(left_line[1] - prev_left[1]) > 35)):
            left = prev_left
    if prev_right:
        if (right_line == None or 
            (right_line[0] > prev_right[0] and right_line[1] < prev_right[1] - 10) or
            (right_line[0] < prev_right[0] and right_line[1] > prev_right[1] + 10) or 
            (not turn and abs(right_line[1] - prev_right[1]) > 35)):
            right = prev_right
    
    return left, right




def detect_lanes(frame_mask):
    global prev_left
    global prev_right
    global slope_border
    global slope_left
    global slope_right
    global middle
    global turn

    relevant_pixels = np.argwhere(frame_mask)

    if skip:
        for i in range(num_samples):
            np.random.choice(points.shape[0], size=2, replace=False)
        return False, None, None

    left_line, right_line = ransac_lines(points=relevant_pixels, 
                                         shoulder_size=shoulder_threshold, 
                                         min_inliers=inliers_threshold,
                                         min_left_slope=slope_left,
                                         max_right_slope=slope_right,
                                         middle_slope=slope_border,
                                         vanishing_point=middle)

    left_line, right_line = verify_lines(left_line=left_line, 
                                         prev_left=prev_left, 
                                         right_line=right_line, 
                                         prev_right=prev_right)

    prev_left = left_line
    prev_right = right_line

    if middle is None:
        middle = (left_line[0] * right_line[1] - right_line[0] * left_line[1]) / (left_line[0] - right_line[0])

    return True, left_line[:2], right_line[:2]


def handle_lane_transition(turn_state, left_line, right_line):
    # detect start of transition 
    # when slope is getting lower
    if not turn and left_line[0] > -1 and right_line[0] > 2:
        turn = -1

    if not turn and right_line[0] < 1 and left_line[0] < -2:
        turn = 1

    # handle transition process
        # when slope goes back to normal (detects new lane line),
        # adjust other lane to detect lines with slope close to 0
    if slope_border == 0:
        # detected new lane line
        if turn == -1 and left_line[0] < -1.2:
            slope_border = prev_left[0] - 0.1
            right_line = prev_left
            slope_right = 2
        elif turn == 1 and right_line[0] > 1.2:
            slope_border = prev_right[0] + 0.1
            left_line = prev_right
            slope_left = -2

    if turn:
        # put transition text
        text = "LEFT" if turn == -1 else "RIGHT"
        cv2.putText(res_frames, text, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 1, cv2.LINE_AA)


    # detect end of transition
        # when both lines have abs(slope) < 1.8
    if turn and left_line[0] > -1.8 and right_line[0] < 1.8:
        slope_border = 0
        slope_left = -3.5
        slope_right = 3.5
        turn = 0


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

    start_time = time()
    np.random.seed(0)

    input_video_path = 'video1.mp4'
    start = 0
    finish = 252
    
    raw_frames, fps = read_frames_from_video(input_video_path)
    print(f'[{time() - start_time:.2f}]')

    final_frames = []
    start_frame = start * fps
    finish_frame = finish * fps
    num_frames = finish_frame - start_frame
    for i, frame in enumerate(raw_frames):
        if i >= finish_frame: break
        skip = i < start_frame

        cropped_frame, top, mask = preprocess_frames(frame)
        valid, left_line, right_line = detect_lanes(pframe, skip)

        if valid:
            
            cropped_frame = draw_line(cropped_frame, right_line)
            cropped_frame = draw_line(cropped_frame, left_line)
            cropped_frame = draw_lane(cropped_frame, left_line, right_line)
            frame[top:, :, :] = cropped_frame

            frame = detect_transition(frame, turn_state, left_line, right_line)

            final_frames.append(pframe)      

        if i % 100 == 0:
            print(f'[{time() - start_time:.2f}] {i}/{num_frames}')
            
    print(f'[{time() - start_time:.2f}]')

    save_frames_to_video(final_frames, fps)
    print(f'[{time() - start_time:.2f}]')
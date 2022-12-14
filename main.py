import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from time import time 

start_time = time()


def print_img(image,name="image"):
    return
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(name)
    plt.show()


def read_frames_from_video(video_path, start, finish):
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
    mid_right = (img.shape[1] - 1, 105)
    mid_left = (0, 105)
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


prev_right = None
prev_left = None
middle = None
turn = 0

def preprocess_frames(raw_frames):
    global prev_left
    global prev_right
    global middle
    global turn

    res_frames = raw_frames
    #print_img(res_frames)
    # Ideas:
        # Crop image
        # Filter only certain color pixels
        # Dilate over lanes to connect broken lines into one lane (- - - -) -> (-------)
        # How to deal with shades (going under bridge)?
            # Normalize colors? 
            # Intensify dark color?
            # Contrast?
    height, width, channel = res_frames.shape
    top = height*62//100

    # crop 
    cropped = res_frames[top:, :, :]
    #print_img(cropped, 'crop')

    # detect edges lines
    blur = cv2.GaussianBlur(cropped, (5,5), 0)
    canny = cv2.Canny(blur, 120, 200)
    #print_img(canny, f'canny')

    # crop polygon
    polygon, poly_mask = poly_crop(canny)
    #print_img(polygon, f'polygon')

    # stay with minimal lines
    squash = squash_lanes(polygon, max_dist=30)
    #print_img(squash, f'squash')

     # RANSAC
    points = np.argwhere(squash)
    d_th = 3
    right_found = False
    left_found = False
    right_line = None
    left_line = None

    for i in range(100):
        p1, p2 = np.random.choice(points.shape[0], size=2, replace=False)
        (y1, x1), (y2, x2) = points[[p1, p2]]
        inliers = []
        inds = []

        for i, (y0, x0) in enumerate(points):
            d = np.linalg.norm(np.cross((x1 - x0, y1 - y0), (x2 - x1, y2 - y1))) / np.linalg.norm((x2 - x1, y2 - y1))
            if d <= d_th:
                inliers.append([y0, x0])
                inds.append(i)

        # focus on lines that fit well in a line
        if len(inliers) < 20: continue

        # fit line to all inliers, but according to inverse dimensions (y - horizontal axis, x - vertical axis)
        inliers = np.array(inliers)

        X = np.concatenate((inliers[:, 0].reshape(-1, 1), np.ones((inliers.shape[0], 1))), axis=1)
        Y = inliers[:, 1].reshape(-1, 1)        
        w, b = np.linalg.lstsq(X, Y, rcond=None)[0].flatten()

        
        # filter slopes in certain range
        if abs(w) > 3.5: continue

        # filter lines according to intersect
        if middle:
            # right lines
            if w > 0 and b < (middle - 20): continue
            # left lines
            if w < 0 and b > (middle + 20): continue

        # right lane
        if w > 0:
            if not right_found or len(inliers) > right_line[2]:
                right_found = True
                right_line = [w, b, len(inliers)]
        # left lane
        else:
            if not left_found or len(inliers) > left_line[2]:
                left_found = True
                left_line = [w, b, len(inliers)]

    # filter lines with bad orientation (accidental lines)
    if prev_left:
        if (left_line == None or 
            (left_line[0] > prev_left[0] and left_line[1] < prev_left[1] - 10) or
            (left_line[0] < prev_left[0] and left_line[1] > prev_left[1] + 10)):
            left_line = prev_left
    if prev_right:
        if (right_line == None or 
            (right_line[0] > prev_right[0] and right_line[1] < prev_right[1] - 10) or
            (right_line[0] < prev_right[0] and right_line[1] > prev_right[1] + 10)):
            right_line = prev_right

    #print(f'left: {left_line}')
    #print(f'right: {right_line}')   

    cropped[:,:,0][poly_mask > 0] = squash[poly_mask > 0]
    cropped[:,:,1][poly_mask > 0] = squash[poly_mask > 0]
    cropped[:,:,2][poly_mask > 0] = squash[poly_mask > 0]

    # TODO: detect start of transition 
        # when slope is getting lower
    if not turn and left_line[0] > -1:
        turn = -1

    if not turn and right_line[0] < 1:
        turn = 1

    # TODO: handle transition process
        # when slope goes back to normal (detects new lane line),
        # adjust other lane to detect lines with slope close to 0
    if turn:
        text = "LEFT" if turn == -1 else "RIGHT"
        cv2.putText(res_frames, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 1, cv2.LINE_AA)

    # TODO: detect end of transition
        # when both lines have abs(slope) < 1.8
    if turn and left_line[0] > -1.8 and right_line[0] < 1.8:
        turn = 0

    cropped = draw_line(cropped, right_line[:2])
    cropped = draw_line(cropped, left_line[:2])
    cropped = draw_lane(cropped, left_line[:2], right_line[:2])
    print_img(cropped, 'cropped')


    prev_left = left_line
    prev_right = right_line

    if middle is None:
        middle = (left_line[0] * right_line[1] - right_line[0] * left_line[1]) / (left_line[0] - right_line[0])

    res_frames[top:, :, :] = cropped
    print_img(res_frames, 'frame')
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
    start = 18
    finish = 83
    raw_frames, fps = read_frames_from_video(input_video_path, start, finish)

    print(f'[{time() - start_time:.2f}]')

    final_frames = []
    num_frames = (finish - start) * fps
    for i, frame in enumerate(raw_frames):
        pframe = preprocess_frames(frame)
        pframe = detect_lanes(pframe)
        final_frames.append(pframe)

        if i % 100 == 0:
            print(f'[{time() - start_time:.2f}] {i}/{num_frames}')
            
    print(f'[{time() - start_time:.2f}]')

    save_frames_to_video(final_frames, fps)
    print(f'[{time() - start_time:.2f}]')
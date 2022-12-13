import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from time import time 

start = time()

class Line:
    def __init__(self, m, n, inliers):
        self.slope = m
        self.y_intercept = n
        self.inliers = inliers
        self.x_intercept = -n / m
        self.num_inliers = len(inliers)
    
    def get_raw_line():
        return [self.slope, self.y_intercept]


def print_img(image,name="image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(name)
    plt.show()


def read_frames_from_video(video_path, start, finish):
    frames = list()
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    

    start_frame = start * fps
    finish_frame = finish * fps
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
    x1 = 0
    y1 = round(m * x1 + n)
    x2 = img.shape[1] - 1
    y2 = round(m * x2 + n)
    return cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
    

def draw_lane(img, left, right):
    lane = img.copy()
    lane[:,:,:] = 0

    ml, nl = left
    mr, nr = right

    top_left = [-nl / ml, 0]
    top_right = [-nr / mr, 0]
    bottom_right = [(img.shape[0] - 1 - nr) / mr, img.shape[0] - 1]
    bottom_left = [(img.shape[0] - 1 - nl) / ml, img.shape[0] - 1]

    points = np.round([[top_left, top_right, bottom_right, bottom_left]]).astype(int)
    cv2.fillPoly(lane, np.round([points]).astype(int), color=[255, 255, 255])
    
    lane_mask = lane > 0    
    img[:, :, 1][lane_mask[:, :, 1]] = 255

    return img


def area_between(line1, line2, imshape):
    (m1, n1), (m2, n2) = line1, line2

    top_left = [-n1 / m1, 0]
    top_right = [-n2 / m2, 0]
    bottom_right = [(imshape[0] - 1 - n2) / m2, imshape[0] - 1]
    bottom_left = [(imshape[0] - 1 - n1) / m1, imshape[0] - 1]

    poly_mask = np.zeros(imshape[:2])
    points = np.round([[top_left, top_right, bottom_right, bottom_left]]).astype(int)
    cv2.fillPoly(poly_mask, points, 1)

    return np.sum(poly_mask)


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
                img[row, mid] = 255
    return img


def poly_crop(img):
    poly_mask = np.zeros_like(img)
    mask_value = 255
    
    top_left = (260, 0)
    top_right = (590, 0)
    bottom_right = (img.shape[1], img.shape[0])
    bottom_left = (0, img.shape[0])

    points = np.array([[top_left, top_right, bottom_right, bottom_left]], dtype=np.int32)
    cv2.fillPoly(poly_mask, points, mask_value)

    return cv2.bitwise_and(img, poly_mask)


prev_right = None
prev_left = None

def preprocess_frames(raw_frames):
    global prev_left
    global prev_right

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

    # detect white lines
    #filtered_frame = cv2.inRange(cropped, np.array([160, 160, 160]), np.array([250, 250, 250]))
    #print_img(filtered_frame, 'filter')
    blur = cv2.GaussianBlur(cropped, (5,5), 0)
    canny = cv2.Canny(blur, 120, 200)
    polygon = poly_crop(canny)

    # stay with minimal lines
    squash = squash_lanes(polygon, max_dist=30)
    #print_img(squash, f'squash')

     # RANSAC
    points = np.argwhere(squash)
    d_th = 1
    right_found = False
    left_found = False
    right_line = None
    left_line = None

    #while not right_found or not left_found:
    for i in range(100):
        p1, p2 = np.random.choice(points.shape[0], size=2, replace=False)
        (y1, x1), (y2, x2) = points[[p1, p2]]
        inliers = []
        inds = []

        for i, (y0, x0) in enumerate(points):
            d = np.linalg.norm(np.cross((x1 - x0, y1 - y0), (x2 - x1, y2 - y1))) / np.linalg.norm((x2 - x1, y2 - y1))
            if d <= d_th:
                inliers.append([x0, y0])
                inds.append(i)

        # focus on lines that fit well in a line
        if len(inliers) < 20: continue

        # fit line to all inliers
        inliers = np.array(inliers)
        X = np.concatenate((inliers[:, 0].reshape(-1, 1), np.ones((inliers.shape[0], 1))), axis=1)
        Y = inliers[:, 1].reshape(-1, 1)        
        w, b = np.linalg.lstsq(X, Y, rcond=None)[0].flatten()

        # filter by slope and x-axis intersect
        if abs(w) < 0.3: continue
        if w > 0 and (-b/w) < ((width * 48) // 100): continue
        if w < 0 and (-b/w) > ((width * 52) // 100): continue

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
            (left_line[0] > prev_left[0] and -left_line[1]/left_line[0] > -prev_left[1]/prev_left[0] + 10) or
            (left_line[0] < prev_left[0] and -left_line[1]/left_line[0] < -prev_left[1]/prev_left[0] - 10)):
            left_line = prev_left
    if prev_right:
        if (right_line == None or 
            (right_line[0] > prev_right[0] and -right_line[1]/right_line[0] > -prev_right[1]/prev_right[0] + 10) or
            (right_line[0] < prev_right[0] and -right_line[1]/right_line[0] < -prev_right[1]/prev_right[0] - 10)):
            right_line = prev_right

    # TODO: detect start of transition 
    # TODO: handle transition process
    # TODO: detect end of transition

    cropped = draw_line(cropped, right_line[:2])
    cropped = draw_line(cropped, left_line[:2])
    cropped = draw_lane(cropped, left_line[:2], right_line[:2])

    prev_left = left_line
    prev_right = right_line

    res_frames[top:, :, :] = cropped
    #print_img(res_frames, 'frame')
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
    raw_frames, fps = read_frames_from_video(input_video_path, 48, 83)

    print(f'[{time() - start:.2f}]')

    preprocessed_frames = [preprocess_frames(frame) for frame in raw_frames]
    final_frames = detect_lanes(preprocessed_frames)
    print(f'[{time() - start:.2f}]')

    save_frames_to_video(final_frames, fps)
    print(f'[{time() - start:.2f}]')
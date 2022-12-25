import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from time import time 


# Previous frame lines
prev_right = None
prev_left = None

# Slope thresholds for line detection
middle_slope = 0
min_left_slope = -3.5
max_right_slope = 3.5

# x coordinate of the meeting point of lane lines (vanishing point)
middle_point = None

# Transition state (left = -1, right = 1, no transition = 0)
transition = 0

# Number of random samples per frame
num_samples = 100

# Thresholds for RANSAC
shoulder_threshold = 1.5
inliers_threshold = 30


def print_img(img, name="image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(name)
    plt.show()


def read_frames_from_video(video_path):
    frames = list()
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    
    num_seconds = num_frames / fps

    print(f"Start reading {num_frames} frames ({num_seconds:.2f} seconds) from {video_path} ({fps} fps)")

    for i in range(num_frames):
        image = capture.read()[1]
        frames.append(image)
    capture.release()

    print(f"Successfully read {len(frames)} frames from {video_path}")

    return frames, fps


def save_frames_to_video(frames, fps, video_path):
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    color = True
    
    writer = cv2.VideoWriter(video_path, codec, fps, (w, h), color)

    print(f"Writing {len(frames)} frames to {video_path} ({fps} fps)")

    for frame in frames:
        writer.write(frame)
    writer.release()


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


def squash_indices(mask, indices, row):
    # Hyperparameters
    num_pixels_right = 1
    num_pixels_left = 0

    mid = (indices[0] + indices[-1]) // 2
    mask[row, indices] = 0
    mask[row, mid - num_pixels_left : mid + num_pixels_right + 1] = 255

    return mask


def squash_edges(edges):
    # Hyperparameter - defines max distance between each pair of neighbor pixels
    max_distance = 40

    for row in range(edges.shape[0]):
        indices = []
        for col in range(edges.shape[1]):
            if edges[row, col] > 0:
                if len(indices) > 0 and col - indices[-1] > max_distance:
                    edges = squash_indices(edges, indices, row)
                    indices = []
                indices.append(col)
        if len(indices) > 0:
            edges = squash_indices(edges, indices, row)
                
    return edges


def polygon_mask(img):
    mask = np.zeros_like(img)
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
    
    cv2.fillPoly(mask, points, mask_value)

    return cv2.bitwise_and(img, mask), mask


def preprocess_frame(raw_frame):
    height, _, _ = raw_frame.shape
    top = height * 62 // 100
    
    # crop 
    cropped = raw_frame[top:, :, :]

    # blur
    blur = cv2.GaussianBlur(cropped, (5,5), 0)

    # detect edges
    canny = cv2.Canny(blur, 100, 200)

    # crop polygon
    polygon, poly_mask = polygon_mask(img=canny)

    # stay with minimal lines
    squash = squash_edges(edges=polygon)

    return cropped, top, squash    


def ransac_lines(points):

    left_line = None
    right_line = None

    for i in range(num_samples):    

        # sample random pair of pixels
        p1, p2 = np.random.choice(points.shape[0], size=2, replace=False)
        (y1, x1), (y2, x2) = points[[p1, p2]]
        inliers = []

        # collect inliers
        for i, (y0, x0) in enumerate(points):
            d = np.linalg.norm(np.cross((x1 - x0, y1 - y0), (x2 - x1, y2 - y1))) / np.linalg.norm((x2 - x1, y2 - y1))
            if d <= shoulder_threshold:
                inliers.append([y0, x0])

        # focus on lines that fit well in a line
        if len(inliers) < inliers_threshold: continue

        # fit line to all inliers, but according to inverse dimensions (y - horizontal axis, x - vertical axis)
        inliers = np.array(inliers)
        X = np.concatenate((inliers[:, 0].reshape(-1, 1), np.ones((inliers.shape[0], 1))), axis=1)
        Y = inliers[:, 1].reshape(-1, 1)        
        w, b = np.linalg.lstsq(X, Y, rcond=None)[0].flatten()

        # filter slopes in certain range
        if w < min_left_slope or w > max_right_slope: continue

        if middle_point:
            # positive slope with low intersect
            if w > 0 and b < (middle_point - 20): continue
            # negative slope with high intersect
            if w < 0 and b > (middle_point + 20): continue

        # select right lane
        if w > middle_slope:
            if right_line is None or len(inliers) > right_line[2]:
                right_line = [w, b, len(inliers)]
        # select left lane
        elif left_line is None or len(inliers) > left_line[2]:
            left_line = [w, b, len(inliers)]

    return left_line[:2] if left_line else None, right_line[:2] if right_line else None


def verify_lines(left_line, right_line):
    left = left_line
    right = right_line

    if prev_left:
        if (left_line == None or 
            
            # new line cannot have smaller slope and bigger intersect, and vice versa
            (left_line[0] > prev_left[0] and left_line[1] < prev_left[1] - 10) or
            (left_line[0] < prev_left[0] and left_line[1] > prev_left[1] + 10) or 
            # new line cannot be too far from previous line
            (not transition and abs(left_line[1] - prev_left[1]) > 35)):

            left = prev_left

    if prev_right:
        if (right_line == None or 

            # new line cannot have smaller slope and bigger intersect, and vice versa
            (right_line[0] > prev_right[0] and right_line[1] < prev_right[1] - 10) or
            (right_line[0] < prev_right[0] and right_line[1] > prev_right[1] + 10) or

            # new line cannot be too far from previous line 
            (not transition and abs(right_line[1] - prev_right[1]) > 35)): 

            right = prev_right
    
    return left, right


def detect_lanes(preprocessed_frame):
    global middle_point

    relevant_pixels = np.argwhere(preprocessed_frame)

    # use RANSAC
    left_line, right_line = ransac_lines(points=relevant_pixels)

    # compare to previous lines
    left_line, right_line = verify_lines(left_line, right_line)

    # init middle point x coordinate (intersection between 2 lines)
    if middle_point is None:
        middle_point = (left_line[0] * right_line[1] - right_line[0] * left_line[1]) / (left_line[0] - right_line[0])

    return True, left_line, right_line


def detect_transition(frame, left_line, right_line):
    global prev_left
    global prev_right
    global transition
    global middle_slope
    global min_left_slope
    global max_right_slope

    new_lane = False
    new_left = left_line
    new_right = right_line

    # detect start of transition - slope is close to 0
    if not transition and left_line[0] > -1 and right_line[0] > 2:
        transition = -1
    if not transition and right_line[0] < 1 and left_line[0] < -2:
        transition = 1

    # handle transition process - capture new lane
    if middle_slope == 0:

        # detect new left line - slope is "suddenly" significantly lower
        if transition == -1 and left_line[0] < -1.2:

            # adjust slope parameters
            middle_slope = prev_left[0] - 0.1
            max_right_slope = 2

            # complete new lane
            new_right = prev_left

        # detect new right line - slope is "suddenly" significantly higher
        elif transition == 1 and right_line[0] > 1.2:

            # adjust slope parameters
            middle_slope = prev_right[0] + 0.1
            min_left_slope = -2

            # complete new lane
            new_left = prev_right

    # detect end of transition - both slopes reach "normal" values
    if transition and left_line[0] > -1.8 and right_line[0] < 1.8:

        # reset slope parameters
        middle_slope = 0
        min_left_slope = -3.5
        max_right_slope = 3.5
        transition = 0

    # draw text
    if transition:
        text = "LEFT" if transition == -1 else "RIGHT"
        frame = cv2.putText(frame, text, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # Save lines for next frame
    prev_left = left_line
    prev_right = right_line 

    return frame, new_left, new_right


if __name__ == '__main__':

    # set random seed
    np.random.seed(0)

    # video parameters
    input_video_path = 'original.mp4'
    output_video_path = 'lanes.mp4'
    
    # read frames
    raw_frames, fps = read_frames_from_video(input_video_path)

    # process frame-by-frame
    final_frames = []
    for i, frame in enumerate(raw_frames):

        # preprocess frame
        cropped_frame, top, preprocessed_frame = preprocess_frame(frame)

        # detect lanes
        relevant, left_line, right_line = detect_lanes(preprocessed_frame)

        # only collect relevant frames
        if relevant:
            
            # detect transition, update lines if new lane captured
            frame, left_line, right_line = detect_transition(frame, left_line, right_line)

            # draw lane
            cropped_frame = draw_line(cropped_frame, left_line)
            cropped_frame = draw_line(cropped_frame, right_line)
            frame[top:, :, :] = draw_lane(cropped_frame, left_line, right_line)    

            # Collect processed frames
            final_frames.append(frame) 

    # save lanes video
    save_frames_to_video(raw_frames, fps, output_video_path)
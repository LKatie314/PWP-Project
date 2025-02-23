import cv2
import numpy as np
import math


# Perspective Transform Setup
def get_perspective_matrices(img_size):
    # Make the perspective transform even more narrow by adjusting the horizontal positions even more
    src = np.float32([
        [img_size[0] * 0.05, img_size[1] * 0.75],  # Even more toward the center
        [img_size[0] * 0.95, img_size[1] * 0.75],  
        [img_size[0] * 0.72, img_size[1] * 0.95],  
        [img_size[0] * 0.28, img_size[1] * 0.95]
    ])


    dst = np.float32([
        [img_size[0] * 0.30, 0],  
        [img_size[0] * 0.70, 0],  
        [img_size[0] * 0.70, img_size[1]],  
        [img_size[0] * 0.30, img_size[1]]
    ])


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def perspective_transform(img, M):
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                cv2.line(line_img, *line, color, thickness)
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)


def calculate_steering_angle(line):
    if line is None:
        return 0
    (x1, y1), (x2, y2) = line
    slope = (y2 - y1) / (x2 - x1)
    angle = np.degrees(np.arctan(slope))
    return angle


def draw_steering_arrow(img, angle):
    arrow_length = 100
    arrow_thickness = 5
    center = (100, 100)
    angle -= 30  # Increase the angle by 15 degrees
    angle_rad = np.radians(angle)
    dx = arrow_length * np.cos(angle_rad)
    dy = arrow_length * np.sin(angle_rad)
    #end_point = (int(center[0] + dx), int(center[1] + dy))
    end_point = (int(center[0] ), int(center[1]-100))
    cv2.arrowedLine(img, center, end_point, (0, 255, 0), arrow_thickness, tipLength=0.5)
    return img


def average_slope_intercept(lines, y_min, y_max):
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []


    if lines is None:
        return None, None


    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # Skip vertical lines
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if abs(slope) < 0.1:  # Filter horizontal lines
                continue
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)


    # Compute weighted averages
    left_lane = None
    if left_weights:
        left_avg = np.dot(left_weights, left_lines) / np.sum(left_weights)
        left_lane = line_points(y_min, y_max, left_avg)
    right_lane = None
    if right_weights:
        right_avg = np.dot(right_weights, right_lines) / np.sum(right_weights)
        right_lane = line_points(y_min, y_max, right_avg)
    return left_lane, right_lane


def line_points(y_min, y_max, line):
    if line is None:
        return None
    slope, intercept = line
    if abs(slope) < 1e-3:
        slope = 1e-3  # Avoid division by zero
    x_min = int((y_min - intercept) / slope)
    x_max = int((y_max - intercept) / slope)
    return ((x_min, int(y_min)), (x_max, int(y_max)))


def draw_centerline(img, left_lane, right_lane):
    if left_lane is None or right_lane is None:
        return img  # If either lane is missing, don't draw a centerline
    (x1_left, y1), (x2_left, y2) = left_lane
    (x1_right, _), (x2_right, _) = right_lane
    x1_center = (x1_left + x1_right) // 2
    x2_center = (x2_left + x2_right) // 2
    centerline = ((x1_center, y1), (x2_center, y2))
    cv2.line(img, centerline[0], centerline[1], (0, 0, 255), 5)
    return img


# Global variables to store the previous frame's lanes
prev_left_lane = None
prev_right_lane = None


def pipeline(image):
    global prev_left_lane, prev_right_lane


    height, width = image.shape[:2]
    img_size = (width, height)
    M, Minv = get_perspective_matrices(img_size)
    warped = perspective_transform(image, M)
    roi_vertices = np.array([[
        (0, height),  
        (width * 0.45, height * 0.6),
        (width * 3, height * 0.6),
        (width * 3, height)
    ]], dtype=np.int32)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    lower_yellow = np.array([10, 0, 100])
    upper_yellow = np.array([4255, 255, 157])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges, roi_vertices)
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )
    y_min = int(height * 0.6)
    y_max = height
    left_lane, right_lane = average_slope_intercept(lines, y_min, y_max)


    # If no lanes are detected, use the previous frame's lanes
    if left_lane is None:
        left_lane = prev_left_lane
    if right_lane is None:
        right_lane = prev_right_lane


    # Update the previous frame's lanes
    prev_left_lane = left_lane
    prev_right_lane = right_lane


    # Calculate steering angle
    steering_angle = calculate_steering_angle(left_lane)


    # Draw final output
    line_image = np.copy(image)
    lanes = []
    if left_lane is not None:
        lanes.append(left_lane)
    if right_lane is not None:
        lanes.append(right_lane)
    result = draw_lines(line_image, lanes)
    result = draw_centerline(result, left_lane, right_lane)
    result = draw_steering_arrow(result, steering_angle)


    return result, warped


def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed, perspective = pipeline(frame)
        display = cv2.resize(processed, (800, 600))
        perspective_display = cv2.resize(perspective, (400, 300))
        combined = np.zeros((600, 800 + 400, 3), dtype=np.uint8)
        combined[:600, :800] = display
        combined[:300, 800:1200] = perspective_display
        cv2.imshow('Lane Detection', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Usage
process_video("C:/Users/walte/Downloads/My Movie.mov")




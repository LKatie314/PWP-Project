"""OVERALL PROGRAM FLOW:

1. SETUP PERSPECTIVE TRANSFORMS:
   - Define source/destination points for road perspective correction
   - Create transformation matrices (M and Minv) for bird's-eye view

2. VIDEO PROCESSING PIPELINE FOR EACH FRAME:
   a. Preprocessing:
      - Resize frame
      - Convert to HLS color space
      - Create masks for white/yellow lane markings
      - Combine masks and apply to original image
      
   b. Edge Detection:
      - Convert to grayscale
      - Gaussian blur
      - Canny edge detection
      - Apply region-of-interest mask

   c. Lane Detection:
      - Hough line transformation to detect line segments
      - Separate left/right lane candidates by slope
      - Average detected lines into single lane boundaries
      - Handle missing frames with previous lane memory

   d. Steering Calculation:
      - Calculate slope between lane points
      - Convert slope to steering angle
      - Special handling for sharp turns (>30 degrees)

   e. Visualization:
      - Draw detected lane boundaries
      - Calculate/draw centerline between lanes
      - Overlay steering direction arrow
      - Perspective transform for bird's-eye view

3. GUI APPLICATION:
   a. Initialize Tkinter Interface:
      - Video display panels (original/processed)
      - Control buttons (play/stop/rewind/ffwd)
      - Logging text area
      - Menu for video selection

   b. Video Playback System:
      - Threaded video processing
      - Frame rate control
      - Time-based steering overrides
      - Aspect ratio-aware resizing
      - Alpha blending for transparent overlays

   c. User Interaction:
      - File dialog for video selection
      - Playback position manipulation
      - Real-time display updates
      - Status logging

4. SPECIAL FEATURES:
   - Dynamic perspective correction
   - Lane memory system for missing detections
   - Time-based steering behavior override
   - Dual video stream display (raw/processed)
   - Responsive GUI with real-time controls

DATA FLOW:
Video File -> Frame Capture -> Color Processing -> Edge Detection ->
Line Detection -> Lane Modeling -> Steering Calculation -> Visualization ->
GUI Display

KEY ALGORITHMS:
- Hough Transform for line detection
- Perspective Warping for road geometry
- Sliding Window for lane averaging
- Alpha Blending for arrow overlay
- Polyfit for curve approximation
"""
import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt


def get_perspective_matrices(img_size):
    """
    Compute perspective transformation matrices for lane detection.

    Args:
        img_size (tuple): Size of the image (width, height).

    Returns:
        tuple: Transformation matrix (M) and inverse transformation matrix (Minv).
    """
    src = np.float32([
        [img_size[0] * 0.05, img_size[1] * 0.75],
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
    """
    Apply perspective transformation to an image.

    Args:
        img (numpy.ndarray): Input image.
        M (numpy.ndarray): Transformation matrix.

    Returns:
        numpy.ndarray: Transformed image.
    """
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))


def skeletonize_image(image_path):
    """
    Skeletonize an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Skeletonized image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary = binary // 255
    skeleton = skeletonize(binary)
    skeleton = (skeleton * 255).astype(np.uint8)
    cv2.imshow('Skeletonized Image', skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return skeleton


def region_of_interest(img, vertices):
    """
    Apply a mask to an image to focus on a region of interest.

    Args:
        img (numpy.ndarray): Input image.
        vertices (numpy.ndarray): Vertices of the polygon defining the region.

    Returns:
        numpy.ndarray: Masked image.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)


def polynomial_fit(x, y, degree):
    """
    Fit a polynomial to the given data points.

    Args:
        x (array-like): Independent variable data points.
        y (array-like): Dependent variable data points.
        degree (int): Degree of the polynomial.

    Returns:
        tuple: Polynomial coefficients and fitted values.
    """
    coeffs = np.polyfit(x, y, degree)
    fitted_values = np.polyval(coeffs, x)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', label='Original Data', marker='o')
    plt.plot(x, fitted_values, color='red', label=f'Polynomial Fit (Degree {degree})', linewidth=2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Polynomial Curve Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()
    return coeffs, fitted_values


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    Draw lines on an image.

    Args:
        img (numpy.ndarray): Input image.
        lines (list): List of lines to draw.
        color (list): Color of the lines.
        thickness (int): Thickness of the lines.

    Returns:
        numpy.ndarray: Image with lines drawn.
    """
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                cv2.line(line_img, *line, color, thickness)
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)


def calculate_steering_angle(line):
    """
    Calculate the steering angle from a line.

    Args:
        line (tuple): Line defined by two points.

    Returns:
        float: Steering angle in degrees.
    """
    if line is None:
        return 0
    (x1, y1), (x2, y2) = line
    slope = (y2 - y1) / (x2 - x1)
    angle = np.degrees(np.arctan(slope))
    return angle


def draw_steering_arrow(img, angle,call):
    """
    Draw a steering arrow on an image.

    Args:
        img (numpy.ndarray): Input image.
        angle (float): Steering angle in degrees.

    Returns:
        numpy.ndarray: Image with steering arrow drawn.
    """
    arrow_length = 100
    arrow_thickness = 5
    center = (100, 100)
    if abs(angle) > 30:
        if angle > 0:
            angle = 0
        else:
            angle = -90
    angle_rad = np.radians(angle)
    dx = arrow_length * np.cos(angle_rad)
    dy = arrow_length * np.sin(angle_rad)
    end_point = (int(center[0] + dx), int(center[1] + dy))
    if call:
        cv2.arrowedLine(img, center, end_point, (0, 255, 0), arrow_thickness, tipLength=0.5)
    else:
        cv2.arrowedLine(img, end_point,center, (0, 255, 0), arrow_thickness, tipLength=0.5)
    return img



def average_slope_intercept(lines, y_min, y_max):
    """
    Average the slope and intercept of detected lines.

    Args:
        lines (list): List of detected lines.
        y_min (int): Minimum y-coordinate.
        y_max (int): Maximum y-coordinate.

    Returns:
        tuple: Left and right lane lines.
    """
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if abs(slope) < 0.1:
                continue
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

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
    """
    Compute the endpoints of a line.

    Args:
        y_min (int): Minimum y-coordinate.
        y_max (int): Maximum y-coordinate.
        line (tuple): Line defined by slope and intercept.

    Returns:
        tuple: Endpoints of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    if abs(slope) < 1e-3:
        slope = 1e-3
    x_min = int((y_min - intercept) / slope)
    x_max = int((y_max - intercept) / slope)
    return ((x_min, int(y_min)), (x_max, int(y_max)))


def draw_centerline(img, left_lane, right_lane):
    """
    Draw a centerline between two lanes.

    Args:
        img (numpy.ndarray): Input image.
        left_lane (tuple): Left lane line.
        right_lane (tuple): Right lane line.

    Returns:
        numpy.ndarray: Image with centerline drawn.
    """
    if left_lane is None or right_lane is None:
        return img
    (x1_left, y1), (x2_left, y2) = left_lane
    (x1_right, _), (x2_right, _) = right_lane
    x1_center = (x1_left + x1_right) // 2
    x2_center = (x2_left + x2_right) // 2
    centerline = ((x1_center, y1), (x2_center, y2))
    cv2.line(img, centerline[0], centerline[1], (0, 0, 255), 5)
    return img


prev_left_lane = None
prev_right_lane = None


def pipeline(image, current_time=None):
    """
    Process an image to detect lanes and compute steering angle.

    Args:
        image (numpy.ndarray): Input image.
        current_time (float): Current timestamp in seconds.

    Returns:
        tuple: Processed image and warped image.
    """
    global prev_left_lane, prev_right_lane

    scale_percent = 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height))

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

    if left_lane is None:
        left_lane = prev_left_lane
    if right_lane is None:
        right_lane = prev_right_lane

    prev_left_lane = left_lane
    prev_right_lane = right_lane

    steering_angle = calculate_steering_angle(left_lane)


    line_image = np.copy(image)
    lanes = []
    if left_lane is not None:
        lanes.append(left_lane)
    if right_lane is not None:
        lanes.append(right_lane)

    result = draw_lines(line_image, lanes)
    result = draw_centerline(result, left_lane, right_lane)
    if current_time is not None and 58 <= current_time <= 62:
        result = draw_steering_arrow(result, steering_angle,False)
    else:
        result = draw_steering_arrow(result, steering_angle,True)

    return result, warped


class VideoApp:
    """
    GUI application for lane detection in videos.
    """

    def __init__(self, root):
        """
        Initialize the VideoApp.

        Args:
            root (tk.Tk): Root window.
        """
        self.root = root
        self.root.title("Lane Detection GUI")
        self.video_path = None
        self.cap = None
        self.is_playing = False

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.overlay_label = tk.Label(root, bg="black")
        self.overlay_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.original_label = tk.Label(root, bg="black")
        self.original_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        control_frame = tk.Frame(root)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.up_button = tk.Button(control_frame, text="Up", command=lambda: self.log("Up pressed"))
        self.up_button.grid(row=0, column=1, padx=10, pady=10)

        self.left_button = tk.Button(control_frame, text="Left", command=self.rewind)
        self.left_button.grid(row=1, column=0, padx=10, pady=10)

        self.stop_button = tk.Button(control_frame, text="Stop", command=self.stop_video)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10)

        self.start_button = tk.Button(control_frame, text="Start", command=self.start_video)
        self.start_button.grid(row=1, column=2, padx=10, pady=10)

        self.right_button = tk.Button(control_frame, text="Right", command=self.fast_forward)
        self.right_button.grid(row=1, column=3, padx=10, pady=10)

        self.down_button = tk.Button(control_frame, text="Down", command=lambda: self.log("Down pressed"))
        self.down_button.grid(row=2, column=1, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(root, width=40, height=10)
        self.log_text.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Video", command=self.open_video)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)

        self.video_path = "C:/Users/walte/Downloads/attempt3.mp4"
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.log("Error: Could not open video file.")
            else:
                self.log(f"Preloaded video: {self.video_path}")

    def open_video(self):
        """Open a video file."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.log("Error: Could not open video file.")
                return
            self.log(f"Opened video: {self.video_path}")

    def start_video(self):
        """Start video playback."""
        if not self.video_path:
            self.log("No video file selected.")
            return
        self.is_playing = True
        threading.Thread(target=self.play_video, daemon=True).start()

    def stop_video(self):
        """Stop video playback."""
        self.is_playing = False
        self.log("Video stopped.")

    def rewind(self):
        """Rewind the video by 1 second."""
        if self.cap is not None:
            current_position = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            new_position = max(0, current_position - 1000)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, new_position)
            self.log(f"Rewinded to {new_position / 1000:.2f} seconds")

    def fast_forward(self):
        """Fast-forward the video by 1 second."""
        if self.cap is not None:
            current_position = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            new_position = current_position + 1000
            self.cap.set(cv2.CAP_PROP_POS_MSEC, new_position)
            self.log(f"Fast-forwarded to {new_position / 1000:.2f} seconds")

    def play_video(self):
        """Play the video with lane detection."""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 4000)
        frame_rate = 15
        delay = int(1000 / frame_rate)

        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                self.log("End of video.")
                break

            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            processed_frame, _ = pipeline(frame, current_time)

            processed_frame = self.crop_and_resize(processed_frame, self.overlay_label.winfo_width(), self.overlay_label.winfo_height())
            original_frame = self.crop_and_resize(frame, self.original_label.winfo_width(), self.original_label.winfo_height())

            self.display_frame(processed_frame, self.overlay_label)
            self.display_frame(original_frame, self.original_label)

            cv2.waitKey(delay)

    def crop_and_resize(self, frame, target_width, target_height):
        """
        Crop and resize a frame to fit target dimensions.

        Args:
            frame (numpy.ndarray): Input frame.
            target_width (int): Target width.
            target_height (int): Target height.

        Returns:
            numpy.ndarray: Cropped and resized frame.
        """
        if frame is None:
            return None
        height, width = frame.shape[:2]
        aspect_ratio = width / height

        if aspect_ratio > (target_width / target_height):
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        resized_frame = cv2.resize(frame, (new_width, new_height))
        background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
        return background

    def display_frame(self, frame, label):
        """
        Display a frame in a Tkinter label.

        Args:
            frame (numpy.ndarray): Input frame.
            label (tk.Label): Label to display the frame.
        """
        if frame is None:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.config(image=imgtk)

    def log(self, message):
        """
        Log a message to the log text area.

        Args:
            message (str): Message to log.
        """
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = VideoApp(root)
    root.mainloop()

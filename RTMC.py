import urllib.request
import cv2
import numpy as np
import datetime
import time
import os
import winsound
import matplotlib.pyplot as plt

images_folder = "captured_images"
report_folder = "report"
image_capture_interval = 5
motion_detection_delay = 30 * 60
image_capture_limit = 200
retry_limit = 10
retry_delay = 5

print("Enter the IP addresses for your cameras. Leave blank to skip an IP.")
ip_camera_urls = []
for i in range(1):
    ip = input(f"Enter IP address for Camera {i} (e.g., http://192.168.x.x:8080/shot.jpg): ").strip()
    if ip:
        ip_camera_urls.append(ip)

os.makedirs(images_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

def is_ip_camera_available(url):
    try:
        response = urllib.request.urlopen(url, timeout=5)
        if response.status == 200:
            return True
    except Exception as e:
        print(f"Failed to connect to IP camera: {e}")
    return False

selected_camera_url = None
cap = None

for url in ip_camera_urls:
    if is_ip_camera_available(url):
        selected_camera_url = url
        print(f"Using IP camera: {selected_camera_url}")
        break

if not selected_camera_url:
    print("No IP cameras available. Checking external USB camera...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("External USB camera not available. Checking built-in webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No cameras available. Exiting.")
            exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
video_filename = f'continuous_recording_{timestamp}.avi'
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
print(f"Recording started: {video_filename}")

last_image_capture_time = 0
image_count = 0
alarm_triggered = False
start_time = time.time()
last_frame = None

while True:
    retry_attempts = 0
    frame = None
    while retry_attempts < retry_limit and frame is None:
        if selected_camera_url:
            response = urllib.request.urlopen(selected_camera_url)
            img_array = np.array(bytearray(response.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)
        else:
            ret, frame = cap.read()

        if frame is None:
            retry_attempts += 1
            print(f"Failed to grab frame. Retrying in {retry_delay} seconds... (Attempt {retry_attempts}/{retry_limit})")
            time.sleep(retry_delay)

    if frame is None:
        print("Frame capture failed after multiple attempts. Exiting program.")
        break

    out.write(frame)

    if time.time() - start_time >= motion_detection_delay:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        if last_frame is None:
            last_frame = blurred_frame
            continue

        frame_diff = cv2.absdiff(last_frame, blurred_frame)
        _, thresh_frame = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        dilated_frame = cv2.dilate(thresh_frame, None, iterations=3)
        contours, _ = cv2.findContours(dilated_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            motion_detected = True
            (cx, cy, cw, ch) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_image_capture_time >= image_capture_interval and image_count < image_capture_limit:
                last_image_capture_time = current_time
                image_filename = os.path.join(images_folder, f'image_capture_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                cv2.imwrite(image_filename, frame)
                image_count += 1

        if image_count >= image_capture_limit and not alarm_triggered:
            winsound.Beep(1000, 2000)
            alarm_triggered = True
            print("Alarm: Image capture limit reached. Motion detection paused.")

        last_frame = blurred_frame

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_datetime, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Motion Capture", frame)
    
    if cv2.waitKey(10) == ord('q'):
        print("Exiting motion capture. Running analysis...")
        break

if cap is not None:
    cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Recording completed: {video_filename}")

training_folder = 'training_images'
testing_folder = images_folder

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def compute_similarity(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is not None and des2 is not None:
        matches = bf.match(des1, des2)
        return len(matches)
    return 0

training_images = []
for file in os.listdir(training_folder):
    img_path = os.path.join(training_folder, file)
    img = cv2.imread(img_path, 0)
    if img is not None:
        training_images.append(img)

results = []
for file in os.listdir(testing_folder):
    test_img_path = os.path.join(testing_folder, file)
    test_img = cv2.imread(test_img_path, 0)
    if test_img is None:
        continue

    total_matches = 0
    for train_img in training_images:
        matches = compute_similarity(train_img, test_img)
        total_matches += matches

    results.append(total_matches)

plt.switch_backend('Agg')
plt.figure(figsize=(10, 5))
plt.bar(range(len(results)), results)
plt.xlabel('Testing Image Index')
plt.ylabel('Malpractice Match Score')
plt.title('Malpractice Detection Analysis')
bar_chart_path = os.path.join(report_folder, 'malpractice_detection_analysis.png')
plt.savefig(bar_chart_path)
print(f"Bar chart saved to: {bar_chart_path}")

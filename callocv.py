import re, io, os, sys, ast, ssl, base64, json, csv, sqlite_utils, asyncio, requests
from datetime import datetime as DT, timezone as TZ, timedelta as TD, time as TIME, date
from sqlite_utils.utils import sqlite3

import cv2, vlc, yt_dlp, threading, time, imutils
import numpy as np

qrcode_detector = cv2.QRCodeDetector()
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
plates_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_license_plate_rus_16stages.xml")

### YouTube Stream URL
def ytStream(url, cookies_path=None):
	ydl_opts = {
		"quiet": True,
		'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best', 
		"coookiefile": "cookies.txt",
	}
	if cookies_path:
		ydl_opts['coookiefile'] = cookies_path
	else:
		ydl_opts['coookiefile'] = "cookies.txt"
	with yt_dlp.YoutubeDL(ydl_opts) as ydl:
		info = ydl.extract_info(url, download=False)
		if 'requested_formats' in info:
			video_url = info['requested_formats'][0]['url']
			audio_url = info['requested_formats'][1]['url']
		else:
			video_url = info['url']
			audio_url = None
			print(f"Video Can't get the audio channel")
		yt_fps = info.get('fps', 30)
		return video_url, audio_url, yt_fps

def snapshort(frame, filePath: None, fileName):
	if not filePath:
		file_save_as = f"{fileName}.jpg"
	else:
		if not os.path.exists(filePath):
			os.makedirs(filePath)
		file_save_as = f"{filePath}/{fileName}.jpg"
	cv2.imwrite(file_save_as, frame)

def play_audio(url):
	if not url:
		return
	player = vlc.MediaPlayer(url)
	player.play()
	return player

def callcamera(resources):
	cap = cv2.VideoCapture()
	if cv2.cuda.getCudaEnabledDeviceCount() > 0:
		cv2.cuda.setDevice(0)
		print(f"CUDA Detected Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
		CUDA_STATUS = True
	else:
		print(f"CUDA NOT Detected Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
		CUDA_STATUS = False
	
	match resources:
		case int():
			cap.open(int(resources))
			cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
			cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		case str():
			if "youtube.com" in resources:
				video_url, audio_url, yt_fps = ytStream(str(resources), 'cookies.txt')
				print(video_url)
				cap.open(str(video_url), cv2.CAP_FFMPEG)
				if audio_url is not None:
					audio_thread = threading.Thread(target=play_audio, args=(str(audio_url), ))
					audio_thread.start()
			else:
				cap.open(str(resources), cv2.CAP_FFMPEG)
		case _:
			print(f"Input Resources Empty. ")
			exit()
	return cap

def ocvcap(cap, brightness_gain):
	if not cap.isOpened():
		return
	ret, frame = cap.read()
	if not ret:
		return
	
	if CUDA_STATUS:
		gpu_frame = cv2.cuda_GpuMat()
		gpu_frame.upload(frame)
		cuda_frame = gpu_frame.download()
	else: pass
	
	brightness_tunner = cv2.convertScaleAbs(cuda_frame if CUDA_STATUS else frame, alpha=1.0 if brightness_gain is None else int(brightness_gain), beta=20)
	mask = object_detector.apply(brightness_tunner)
	gray_image = cv2.cvtColor(brightness_tunner, cv2.COLOR_BGR2GRAY)
	
	return brightness_tunner, mask, gray_image

def corecv(cap, brightness_gain, label_border, min_area, max_area):
	ref, frame = cap.read()
	width, height, fps = frame.shape
	if not ref:
			if cap : cap.release()
			print("Failed to grab frame!")
			return None
	if cv2.cuda.getCudaEnabledDeviceCount() > 0:
		gpu_frame = cv2.cuda_GpuMat()
		gpu_frame.upload(frame)
		cuda_frame = gpu_frame.download()
		CUDA_STATUS = True
	else: 
		CUDA_STATUS = False

	cv2.ocl.setUseOpenCL(True)
	brightness_tunner = cv2.convertScaleAbs(cuda_frame if CUDA_STATUS else frame, alpha=1.0 if brightness_gain is None else int(brightness_gain), beta=20)
	mask = object_detector.apply(brightness_tunner)
	gray_image = cv2.cvtColor(brightness_tunner, cv2.COLOR_BGR2GRAY)

	qrcodeDetection(brightness_tunner if not CUDA_STATUS else cuda_frame)
	motionDetection(brightness_tunner if not CUDA_STATUS else cuda_frame, mask, label_border, min_area, max_area)
	faceDetection(brightness_tunner if not CUDA_STATUS else cuda_frame, gray_image, label_border)
	wordsDetection(brightness_tunner if not CUDA_STATUS else cuda_frame, gray_image)

	return brightness_tunner, mask, gray_image, CUDA_STATUS

### Face Detection
def faceDetection(frame, gray_image, label_border):
	face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
 
	if len(face) > 0:
		x, y, w, h = face[0]
		bbox = (x, y, h, w)
		try:
			tracker = cv2.legacy.TrackerCSRT_create()
			tracker.init(frame, bbox)
		except AttributeError:
			print("Error: CSRT tracker not available. Ensure opencv-contrib-python is installed.")
			return
		# timestamp = DT.now().strftime("%Y-%m-%d_%H-%M-%S")
		# snapshort(frame, filePath=None, fileName=timestamp)
		# print(f"Saved snapshot: {timestamp}.jpg")
 
	for (x, y, w, h) in face:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
		cv2.putText(frame, "FACE DETECT", (x, y - int(label_border)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def motionDetection(frame, mask, label_border, min_area, max_area):
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		area = cv2.contourArea(contour)
		if int(min_area) < area < int(max_area):
			(x, y, w, h) = cv2.boundingRect(contour)
			cv2.rectangle(frame, (x - int(label_border), y - int(label_border)), (x + w + int(label_border), y + h + int(label_border)), (0, 255, 0), 1)
			cv2.putText(frame, "DETECT", (x- int(label_border), y - int(label_border)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
		elif 2000 < area < 2800:
			(x, y, w, h) = cv2.boundingRect(contour)
			cv2.rectangle(frame, (x - int(label_border), y - int(label_border)), (x + w + int(label_border), y + h + int(label_border)), (0, 255, 255), 1)
			cv2.putText(frame, "DETECT", (x- int(label_border), y - int(label_border)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
		else:
			continue

### QR Code Detection
def qrcodeDetection(frame):
	data, points, _ = cv2.QRCodeDetector().detectAndDecode(frame)
	if points is not None:
		points = points[0]
		points = [(int(point[0]), int(point[1])) for point in points]
		
		for i in range(len(points)):
			pt1 = points[i]
			pt2 = points[(i+1) % len(points)]
			cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 0), 1)
		if data:
			qr_content = data
			print(qr_content)

### Words Detection
def wordsDetection(frame, gray_image):
	plates = plates_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	for (x,y,w,h) in plates:
		plates_rec = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 1)        
		cv2.putText(plates_rec, 'Text', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

		# fileName = f"{DT.now().strftime('%Y-%m-%d_%H-%M')}"
		# snapshort(frame, None, fileName)

############################
### You can copy the following code to use in your project
### to reacall your cv from function
### * StreamViewer : You need to follow the rules to give the values of function,
### * cap : call camera from functoin, you can also <callcamera> from the other functoin, 
### 				This is include the CUDA checking, and will return the Video Capture with src.
### * corecv : Real OpenCV Frame filter, you must make the detection in the <corecv> function
### ** The Demo are show in __main__, update the info for test your devices.
############################
############################COPY THIS CODE TO YOUR PROJECT############################>
### MAIN Function
def StreamViewer(camera_src, brightness_gain, label_border, min_area, max_area):
	cap = callcamera(camera_src)

	if not cap.isOpened():
		print(f"Failed to open camera source: {camera_src}")
		exit()
	
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	print(f"Camera Resolution: {width}x{height}, FPS: {fps}")
	### SAVE STREAM to MP4
	#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#temp_video_path = 'temp_video.mp4'
	#out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
	
	while True:
		frame, mask, gray_image = corecv(cap, brightness_gain, label_border, min_area, max_area)
		if frame is None:
			print("Failed to grab frame!")
			break
	
		# out.write(frame)  # Save the frame to the video file
		MainFrameViewerName, MaskFrameViewerName = "VIEWER", "MASK"
		cv2.namedWindow(f"{MainFrameViewerName} - (GPU)" if CUDA_STATUS else f"{MainFrameViewerName} - (CPU)", cv2.WINDOW_NORMAL)
		cv2.imshow(f"{MainFrameViewerName} - (GPU)" if CUDA_STATUS else f"{MainFrameViewerName} - (CPU)", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
############################COPY THIS CODE TO YOUR PROJECT############################>

if __name__ == "__main__":
	username, password, ip, channel = "ExampleUserName", "ExamplePassword", "192.168.xxx.XXX", 4
	path = f"channel={int(channel)}&stream=0.sdp"
	canera_src = f"rtsp://{username}:{password}@{ip}/{path}"
	label_border, brightness_gain = 12, 1.0
	min_area, max_area = 450, 900
	StreamViewer(canera_src, brightness_gain, label_border, min_area, max_area)

############################
### callcamera : check CUDA and Canera Source, make it reture the VideoCapture object
### corecv : Core of the OpenCV frame filter, Decetion must be in <corecv>
### play_audio: Play audio file by Python-VLC
### snapshort: Take a snapshot from the video stream (<frame>, <filePath>: Default=None, <fileName>[No need to put {.mp4} at the end.])
### motionDetection: Detect motion in the video stream (<frame>, <brightness_gain>, <label_border>, <min_area>, <max_area>)
### qrcodeDetection: Detect QR codes in the video stream (<frame>), "snapshot" = Disable Default
### faceDetection: Detect faces in the video stream (<frame>) "snapshot" = Disable Default
### wordsDetection: Detect words in the video stream
### out : Save the stream to MP4 format, you can change the save format in <fourcc>
### ==========================
### object_detector : using createBackgroundSubtractorMOG2()
### face_classifier : OpenCV face detection basic model. [haarcascade_frontalface_default.xml]
### words_classifier : OpenCV text detection basic model. [haarcascade_russian_plate_number.xml]
### CUDA_STATUS : Default False to disable, True to enable
### === pip install ===
### python3 -m pip install -U opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
############################

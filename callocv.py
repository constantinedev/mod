import re, os, io, sys, ast, ssl, json, csv, sqlite_utils, requests, asyncio, time, pgpy
from datetime import datetime as DT, timezone as TZ, timedelta as TD, time as TIME, date
from sqlite_utils.utils import sqlite3

import cv2, vlc, threading, yt_dlp, imutils, qrcode, pyotp, jwt

object_detector = cv2.createBackgroundSubtractorKNN()
qrcode_detector = cv2.QRCodeDetector()
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
plates_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_license_plate_rus_16stages.xml')

### Play Audio
def play_audio(url):
	if not url:
		return
	player = vlc.MediaPlayer(url)
	player.play()
	return player

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

async def snapshort(frame, fileName, filePath : None):
	if not filePath:
		file_save_as = f"temps/images/cvcap/{fileName}.jpg"
	else:
		file_save_as = f"{filePath}/{fileName}.jpg"
	cv2.imwrite(file_save_as, frame)
	return

CUDA_STATUS = False
### OpenCV Core
def streaming(camera_src, brightness_gain, label_border, min_area, max_area):
	if cv2.cuda.getCudaEnabledDeviceCount() > 0:
		cv2.cuda.setDevice(0)
		CUDA_STATUS = True
		print(f"CUDA Device(s) Detected! {cv2.cuda.getCudaEnabledDeviceCount()}")
	else:
		CUDA_STATUS = False
		print(f"CUDA Device(s) Not Detected! {cv2.cuda.getCudaEnabledDeviceCount()}, CPU are running...")
	
	match camera_src:
		case int():
			cap = cv2.VideoCapture(int(camera_src), cv2.CAP_DSHOW)
		case str():
			if 'youtube.com' in camera_src:
				video_url, audio_url, yt_fps = ytStream(str(camera_src), cv2.CAP_FFMPEG)
				cap = cv2.VideoCapture(str(video_url, cv2.CAP_FFMPEG))
				if audio_url:
					audio_thread = threading.Thread(target=play_audio, args=(str(audio_url), ))
					audio_thread.start()
			else:
				cap = cv2.VideoCapture(str(camera_src), cv2.CAP_FFMPEG)
		case _:
			print(f"Camera Not get Resources.")
			return f"Camera Not get Resources."
	
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(width, height, fps, counts)
	
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	temp_video_path = 'temp_video.mp4'
	out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
	
	if not cap.isOpened():
		if cap : cap.release()
		if out : out.release()
		print(f"Camera not Connected!")
	
	while True:
		frame, mask = ocvcore(cap, brightness_gain, label_border, min_area, max_area)
		if frame is None:
			break

		### DEBUG FOR SHOWING TEH FRAME
		cv2.imshow("FRAME SHOW", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
### OpenCV Core [END]

def ocvcore(cap, brightness_gain, label_border, min_area, max_area):
	ret, frame = cap.read()
	width, height, fps = frame.shape
	if not ret:
		return None, None

	if brightness_gain is None:
		brightness_gain = 1.0
	else:
		brightness_gain = int(brightness_gain)
	bright_tuner = cv2.convertScaleAbs(frame, alpha=brightness_gain, beta=0)

	if CUDA_STATUS:
		gpu_frame = cv2.cuda_GpuMat()
		gpu_frame.upload(bright_tuner)
		cuda_frame = gpu_frame.download()
	else: pass
	
	frame = cuda_frame if CUDA_STATUS else bright_tuner
	mask = object_detector.apply(frame)
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	qrcodeDetect(frame)
	motionDetection(frame, mask, label_border, min_area, max_area)
	faceDetection(frame, gray_image, label_border)
	wordsDecetion(frame, gray_image)
	
	return frame, mask

### FACE DETECTION
def faceDetection(frame, gray_image, label_border):
	face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	if len(face) > 0:
		x, y, w, h = face[0]
		bbox = (x, y, h, w)
		try:
			tracker = cv2.legacy.TrackerCSRT_create()
			tracker.init(frame, bbox)
			
			# timestamp = DT.now().strftime("%Y-%m-%d_%H-%M-%S")
			timestamp = DT.now().strftime("%Y-%m-%d_%H-%M")
			fileName = timestamp
			# snapshort(frame, fileName, None)
		except AttributeError:
			print("Error: CSRT tracker not available. Ensure opencv-contrib-python is installed.")
			return

	for (x, y, w, h) in face:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
		cv2.putText(frame, "FACE DETECT", (x, y - int(label_border)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

### MOTION DETECT
def motionDetection(frame, mask, label_border, min_area, max_area):
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		area = cv2.contourArea(contour)
		if int(min_area) < area < int(max_area):
			(x, y, w, h) = cv2.boundingRect(contour)
			cv2.rectangle(frame, (x - int(label_border), y - int(label_border)), (x + w + int(label_border), y + h + int(label_border)), (0, 255, 0), 1)
			cv2.putText(frame, "DETECT", (x - int(label_border), y - int(label_border)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
			# timestamp = DT.now().strftime("%Y-%m-%d_%H-%M")
			# fileName = timestamp
			# snapshort(frame, fileName, None)
		# elif 1800 < area < 2000:
		# 	(x, y, w, h) = cv2.boundingRect(contour)
		# 	cv2.rectangle(frame, (x - int(label_border), y - int(label_border)), (x + w + int(label_border), y + h + int(label_border)), (0, 255, 255), 1)
		# 	cv2.putText(frame, "DETECT", (x - int(label_border), y - int(label_border)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
		else:
			continue

### QRCode Detect
def qrcodeDetect(frame):
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
			# snapshort(frame, "testsave.jpg", None)
			# return qr_content
			### Make Output Data to NEXT step

def wordsDecetion(frame, gray_image):
	plates = plates_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
	for (x,y,w,h) in plates:
		plates_rec = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)        
		cv2.putText(plates_rec, 'Text', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

		# fileName = f"{DT.now().strftime('%Y-%m-%d_%H-%M')}"
		# snapshort(frame, fileName, None)

# loop = asyncio.get_event_loop()
# loop.run_until_complete(streaming())

### Python Version Request: up to 3.11
### Modueles requests: opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 
### Options requests: CUDA
### Descriptions: craete your cap to ocvcore and get frame and mask with other detections. 
### streaming: is also the debug when you given the drightness, boxs border, min aree, max area for showing with imshow() for debug.
### If you want to use the modules with a copy you can edit in "streaming" for development your funciton.

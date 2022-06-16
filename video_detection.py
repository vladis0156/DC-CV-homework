import cv2
import torch
import os


# Модель yolov5
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model.cuda()

# Получение кадров из видеоряда
video_capture = cv2.VideoCapture('example_video.mp4')
video_capture.set(cv2.CAP_PROP_FPS, 1 / 25)

saved_frame_name = 0
while video_capture.isOpened():
    frame_is_read, frame = video_capture.read()

    if frame_is_read:
        cv2.imwrite(f"frames/{saved_frame_name}.jpg", frame)
        saved_frame_name += 1

    else:
        print("Frames received!")
        break

# Получение кадров с баундбоксами
frame_count = 0
for i in range(980):
    im = f'./frames/{frame_count}.jpg'
    frame_count += 1
    result = model(im, size=640)
    result.print()
    result.save(save_dir='./modified_frames')

os.system('ffmpeg -r 25 -i ./modified_frames/%d.jpg result.mpg')

print('Done!')

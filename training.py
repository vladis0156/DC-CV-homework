import torch
import json
import os

# Модель yolov5
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model.cuda()

# обработка .json и запись .txt файлов
files = os.listdir('./json labels')
for file in files:
    with open(f'./json labels/{file}', 'r') as read_file:
        data = json.load(read_file)
        image_number = file.split('.')[0]
        if f'{image_number}.txt' in os.listdir('dataset/labels'):
            os.remove(f'dataset/labels/{image_number}.txt')
    for i in range(len(data['shapes'])):
        im_height, im_width = ((data['imageHeight']), (data['imageWidth']))
        coordinates = data['shapes'][i]['points']
        class_number = '0'
        x1, y1, x2, y2 = ((coordinates[0][0]),
                          (coordinates[0][1]),
                          (coordinates[1][0]),
                          (coordinates[1][1]))
        # Нормализация координат бб
        h = abs(y1 - y2)
        w = abs(x1 - x2)
        xc = (min(x1, x2) + w / 2) / im_width
        yc = (min(y1, y2) + h / 2) / im_height
        lst = [class_number, str(xc), str(yc), str(w / im_width), str(h / im_height)]
        data_string = ' '.join(lst)
        with open(f'dataset/labels/{image_number}.txt', 'a+') as write_file:
            write_file.write(data_string + '\n')

# Обучение модели
os.system('python ./yolov5/train.py --img 640 --batch 2 --epochs 10 --data custom_dataset.yaml --weights ./yolov5l.pt --name result')

# Проверка обученной модели
os.system('python ./yolov5/detect.py --img 640 --weights ./yolov5/runs/train/result/weights/best.pt --conf 0.15 --source ./dataset/images --name result_detect --save-txt')

print('Done!')

import os

import boto3
import cv2

import credentials


output_dir = 'data'
output_dir_imgs = os.path.join(output_dir, 'images')
output_dir_anns = os.path.join(output_dir, 'anns')

# create AWS Reko client
reko_client = boto3.client('rekognition', region_name='us-west-2',
                           aws_access_key_id=credentials.access_key,
                           aws_secret_access_key=credentials.secret_key)

# set the target class
target_class = 'Gun'

# load video
cap = cv2.VideoCapture('data/gun.mp4')

frame_nmr = -1

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    H, W, _ = frame.shape

    # convert frame to jpg
    _, buffer = cv2.imencode('.jpg', frame)

    # convert buffer to bytes
    image_bytes = buffer.tobytes()

    # detect objects
    response = reko_client.detect_labels(Image={'Bytes': image_bytes},
                                             MinConfidence=50)

    for label in response['Labels']:
        if label['Name'] == target_class:
            for instance_nmr in range(len(label['Instances'])):
                bbox = label['Instances'][instance_nmr]['BoundingBox']
                x1 = int(bbox['Left'] * W)
                y1 = int(bbox['Top'] * H)
                width = int(bbox['Width'] * W)
                height = int(bbox['Height'] * H)
                cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 3)
                # Добавление текста над квадратом
                text = target_class
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 7
                color = (255, 255, 255)  # Белый цвет
                thickness = 6
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x1 + (width - text_size[0]) // 2  # Центрирование текста по горизонтали
                text_y = y1 - 5  # Позиция над квадратом

                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    new_width = 640
    new_height = 480
    frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



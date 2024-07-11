# Обнаружение оружия в видео с использованием AWS Rekognition и OpenCV
![Screen](https://github.com/Asshkid/AWS-recognition/assets/132083258/5cc1456c-f180-476a-9bc1-003fc4a380a0)

Этот проект предназначен для обнаружения оружия в видеопотоке с использованием AWS Rekognition и OpenCV. Кадры видео обрабатываются для идентификации и аннотирования оружия, предоставляя визуальную обратную связь через рамки и метки.

## Требования

- Python 3.x
- OpenCV
- Boto3
- AWS учетные данные с разрешениями на использование Rekognition

## Установка

1. Клонируйте репозиторий:
    ```bash
    git clone https://github.com/yourusername/gun-detection
    ```

2. Перейдите в директорию проекта:
    ```bash
    cd gun-detection
    ```

3. Установите необходимые зависимости:
    ```bash
    pip install opencv-python boto3
    ```

4. Убедитесь, что у вас есть файл `credentials.py` с вашими учетными данными AWS:
    ```python
    access_key = 'your_access_key'
    secret_key = 'your_secret_key'
    ```

## Использование

1. Поместите видеофайл, который вы хотите использовать для обнаружения оружия, в директорию `data` и назовите его `gun.mp4`.

2. Запустите скрипт:
    ```bash
    python detect_gun.py
    ```

## Описание скрипта

Скрипт выполняет следующие действия:

1. Создает клиента AWS Rekognition.
2. Загружает видеофайл.
3. Читает и обрабатывает кадры видео.
4. Отправляет каждый кадр в AWS Rekognition для обнаружения объектов.
5. Если обнаружено оружие, рисует рамку вокруг него и добавляет метку.
6. Изменяет размер кадра для отображения.
7. Показывает обработанный кадр.
8. Останавливает выполнение, если нажата клавиша 'q'.

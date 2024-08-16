import os
import cv2
import time
import numpy as np
import argparse
import math

from src.hand_tracking import HandDetector
from src.config import *


def get_args():
    parser = argparse.ArgumentParser(description="Data Collection")
    parser.add_argument("--data_path", "-d", type=str, default="../dataset/alphabet")
    parser.add_argument("--start_image", "-s", type=int, default=0)
    parser.add_argument("--num_image", "-n", type=int, default=500)
    parser.add_argument("--image_size", "-i", type=int, default=300)

    args = parser.parse_args()
    return args


def main(args):
    # create data folder
    data_path = args.data_path
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
        for item in CLASSES:
            os.makedirs(os.path.join(data_path, item))

    # input
    print("Classes = {}".format(CLASSES))
    while True:
        input_alphabet = input("Input class you want to collect data: ")
        if input_alphabet in CLASSES:
            print("Opening camera. Please wait...")
            break
        else:
            print("Error! Please input again.")

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    start_time = time.time()
    while True:
        ret, image = cap.read()
        hands, image = detector.findHands(image)
        elapsed_time = time.time() - start_time

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((args.image_size, args.image_size, 3), np.uint8) * 255
            imgCrop = image[y - 20:y + h + 20, x - 20:x + w + 20]

            aspecRatio = h / w
            if aspecRatio > 1:
                n = args.image_size / h
                wCal = math.ceil(n * w)
                imgResize = cv2.resize(imgCrop, (wCal, args.image_size))
                wGap = math.ceil((args.image_size - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                n = args.image_size / w
                hCal = math.ceil(n * h)
                imgResize = cv2.resize(imgCrop, (args.image_size, hCal))
                hGap = math.ceil((args.image_size - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # cv2.imshow('Image Crop', imgCrop)
            # cv2.imshow('Image White', imgWhite)

        if elapsed_time < 20:
            cv2.putText(image, "Press ""s"" to collect image for {} class".format(input_alphabet),
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, WHITE, 2)
        cv2.imshow('Data Collector', image)

        # Save data
        count = args.start_image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            count += 1
            image_name = str(time.time())
            cv2.imwrite("{}/{}/{}.jpg".format(data_path, input_alphabet, image_name), imgWhite)
            print('Saving image no{}: {}'.format(count, image_name))
            if count == args.num_image:
                break

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    main(args)

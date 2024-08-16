import cv2
import math
import numpy as np
import argparse

from src.hand_tracking import HandDetector
from src.classification import Classifier
from src.config import *
from src.utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Inferent Sign Language Detection")
    parser.add_argument("--image_size", "-i", type=int, default=300)
    parser.add_argument("--checkpoint", "-p", type=str, default="trained_models/last.pt")

    args = parser.parse_args()
    return args


def inference(args):
    prediction = []
    sentence = []

    # camera
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    classifier = Classifier(args.checkpoint)

    while True:
        ret, image = cap.read()

        # Detector
        hands, image = detector.findHands(image)

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

                results, index = classifier.prediction(imgWhite)
                prediction.append(np.argmax(results))
                if most_common_value(prediction[-25:]) == np.argmax(results):
                    character = special_characters_prediction(sentence, CLASSES[index])
                    sentence.append(character) if not sentence or sentence[-1] != character else None
            else:
                n = args.image_size / w
                hCal = math.ceil(n * h)
                imgResize = cv2.resize(imgCrop, (args.image_size, hCal))
                hGap = math.ceil((args.image_size - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

                results, index = classifier.prediction(imgWhite)
                prediction.append(np.argmax(results))
                if most_common_value(prediction[-25:]) == np.argmax(results):
                    character = special_characters_prediction(sentence, CLASSES[index])
                    sentence.append(character) if not sentence or sentence[-1] != character else None

            # Visualize
            cv2.rectangle(image, (x - 20, y - 20 - 50), (x - 20 + 90, y - 20 - 50 + 50), MAGENTA, cv2.FILLED)
            cv2.putText(image, character, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, WHITE, 2)
            cv2.rectangle(image, (x - 20, y - 20), (x + w + 20, y + h + 20), MAGENTA, 4)

        # cv2.imshow('Image Crop', imgCrop)
        # cv2.imshow('Image White', imgWhite)
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    inference(args)

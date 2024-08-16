import cv2
import mediapipe as mp

from src.config import *


class HandDetector:
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxHands,
                                        self.model_complexity,
                                        self.detectionCon,
                                        self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.tips = [4, 8, 12, 16, 20]

    def findHands(self, image, flipType=True, draw=True, color=RED):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)
        # print(self.results.multi_handedness)

        allHands = []
        h, w, c = image.shape
        if self.results.multi_hand_landmarks:
            for handLm, handType in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                myHand = {}
                lmList = []
                xList = []
                yList = []
                # landmark list
                for idx, landmark in enumerate(handLm.landmark):
                    px, py, pz = int(landmark.x * w), int(landmark.y * h), int(landmark.z * w)
                    lmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                # bounding box
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                width, height = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, width, height

                # hand information
                myHand["lmList"] = lmList
                myHand["bbox"] = bbox

                # hand type
                if handType:
                    if flipType:
                        if handType.classification[0].label == "Right":
                            myHand["Type"] = "Left"
                        else:
                            myHand["Type"] = "Right"
                    else:
                        myHand["Type"] = handType.classification[0].label
                allHands.append(myHand)

                # visualize landmark and bounding box
                if draw:
                    self.mpDraw.draw_landmarks(image, handLm, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(image,
                                  (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  color, 2)

        return allHands, image


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        ret, frame = cap.read()
        hands, image = detector.findHands(frame)
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            handType1 = hand1["Type"]
        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            handType2 = hand2["Type"]

        cv2.imshow("Image", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    main()

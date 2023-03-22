import cv2
import mediapipe as mp

def HandTracking(url_wideo, url_zdjecie):
    mode = input("Wybierz tryb (w - wideo, z - zdjęcie): ")

    if mode == 'w':
        cap = cv2.VideoCapture(url_wideo)
    elif mode == 'z':
        img = cv2.imread(url_zdjecie)


    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0
    threshold = 0.2 # wartość progowa

    while True:
        if mode == 'w':
            success, img = cap.read()
            img = cv2.flip(img, 1)  # obrocony obraz z kamery
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            total_finger_count = 0
            hand_count = len(results.multi_hand_landmarks)
            for handLms in results.multi_hand_landmarks:
                finger_count = 0
                hand_center = handLms.landmark[0]
                for id, lm in enumerate(handLms.landmark):
                    if id in [0, 4, 8, 12, 16, 20, 24]:  # tylko te punkty zostaną narysowane
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 0:  # punkt 0 jest umieszczony w środku dłoni
                            continue
                        distance = ((hand_center.x - lm.x) ** 2 + (hand_center.y - lm.y) ** 2) ** 0.5
                        if distance > threshold:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                            finger_count += 1
                total_finger_count += finger_count
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            cv2.putText(img, "Fingers: " + str(total_finger_count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if hand_count == 2:
                cv2.putText(img, "Both hands", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif hand_count == 1:
                if mode == "w":
                    if results.multi_hand_landmarks[0].landmark[0].x < 0.5:
                        cv2.putText(img, "Left hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, "Right hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif mode == "z":
                    if results.multi_hand_landmarks[0].landmark[0].x < 0.5:
                        cv2.putText(img, "Right hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, "Left hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Hands detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if mode == 'w':
        cap.release()
    cv2.destroyAllWindows()


HandTracking(0,"examples/hand.jpg")

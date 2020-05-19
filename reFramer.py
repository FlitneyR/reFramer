import cv2, sys

# faceCascade = cv2.CascadeClassifier(cascPath)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

lastFacePos = None

vPadding = 1.5 # three faces
hPadding = 16 * vPadding / 9

damping = 9

IMG_W = 1280
IMG_H = 720

while True:
    ret, frame = video_capture.read()

    if lastFacePos == None:
        lastFacePos = [0, 0, frame.shape[1], frame.shape[0]]

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    facePos = None

    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if facePos == None:
            facePos = [x, y, w, h]
        elif w * h > facePos[2] * facePos[3]:
            facePos = [x, y, w, h]

    if facePos == None:
        # facePos = [0, 0, frame.shape[1], frame.shape[0]]
        facePos = lastFacePos

    for i in range(len(facePos)):
        facePos[i] = (facePos[i] + damping * lastFacePos[i]) / (damping + 1)

    x, y, w, h = facePos

    x = x + w / 2
    y = y + h / 2

    s = (w + h) / 2

    top     = int(max(y - vPadding * s, 0))
    bottom  = int(min(y + vPadding * s, frame.shape[0]))
    left    = int(max(x - hPadding * s, 0))
    right   = int(min(x + hPadding * s, frame.shape[1]))

    crop_img = frame[top:bottom, left:right]

    resized = cv2.resize(crop_img, (IMG_W, IMG_H), cv2.INTER_CUBIC)

    cv2.imshow('Video', resized)

    lastFacePos = facePos

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

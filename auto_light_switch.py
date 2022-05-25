import cv2
import numpy
import os
import asyncio
from kasa import SmartPlug
from kasa import Discover

async def getLights():
    devices = await Discover.discover()
    print(devices)
    if not len(devices):
        print("No devices found")
        quit()

    switchIP = next(iter(devices))
    return switchIP

absolute_path = os.path.abspath(__file__)
directoryPath = os.path.dirname(absolute_path)

def detect(frame, faceCascade, profileCascade):
    h, w, c = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    box_coordinates_left_profile =  profileCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    box_coordinates_right_profile = profileCascade.detectMultiScale(cv2.flip(frame,1), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    box_coordinates_face = faceCascade.detectMultiScale(cv2.flip(frame,1), scaleFactor=1.1, minNeighbors=12, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bounding_box_cordinates = numpy.array([])
    if (len(box_coordinates_left_profile)): bounding_box_cordinates = box_coordinates_left_profile
    elif (len(box_coordinates_right_profile)): 
        bounding_box_cordinates = box_coordinates_right_profile
        bounding_box_cordinates[:, 0] = w - bounding_box_cordinates[:, 0] - 1 - bounding_box_cordinates[:, 2] # reflect the box horizontally
        
    elif (len(box_coordinates_face)): bounding_box_cordinates = box_coordinates_face

    count = 0
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
    
    cv2.imshow('output', frame)

    # print("Number of people detected: " + str(count))

    return count > 0


async def detectByCamera(faceCascade, profileCascade, switchIP):   
    switch = SmartPlug(switchIP)
    video = cv2.VideoCapture(0)

    await switch.turn_on()
    lightsOn = True
    lightsOnThreshold = 10
    lightsOffThreshold = 10
    lightsOnCount = 0
    lightsOffCount = 0

    while True:
        check, frame = video.read()
        detected = detect(frame, faceCascade, profileCascade)

        if (detected):
            lightsOnCount += 1
        else:
            lightsOffCount += 1

        if (lightsOnCount > lightsOnThreshold):
            if not lightsOn:
                print("Turning on light...")
                await switch.turn_on()
            else:
                print("Light already on")

            lightsOn = True
            lightsOnCount = 0
            lightsOffCount = 0

        if (lightsOffCount > lightsOffThreshold):
            if lightsOn:
                print("Turning off light...")
                await switch.turn_off()
            else:
                print("Light already off")
            lightsOn = False
            lightsOnCount = 0
            lightsOffCount = 0

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    switchIP = asyncio.run(getLights())

    profileCascadeModel = directoryPath + "\haarcascade_profileface.xml"
    faceCascadeModel = directoryPath + "\haarcascade_frontalface_default.xml"

    profileCascade = cv2.CascadeClassifier(profileCascadeModel)
    faceCascade = cv2.CascadeClassifier(faceCascadeModel)

    asyncio.run(detectByCamera(faceCascade, profileCascade, switchIP))


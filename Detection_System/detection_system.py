import cv2
import csv
import mediapipe as mpp
import moviepy.editor as mp
import speech_recognition as sr
from gaze_tracking import GazeTracking
from fer import FER
from fer import Video

level = "Demo"

i = 4
while i <= 6:

    blink_count = 0
    pose_count = 0
    gaze_direction = ""
    gaze_direction_number = 0
    # 1 = robot, 2 = tablet, 3 = away
    stage = ""
    mp_drawing = mpp.solutions.drawing_utils
    mp_hands = mpp.solutions.hands

    video_location = "/Users/isaacroberts/Desktop/Engagement_Videos/" + level + "_Engagement/"+str(i)+".m4v"
    cap = cv2.VideoCapture('/Users/isaacroberts/Desktop/Engagement_Videos/' + level + '_Engagement/'+str(i)+'.m4v')
    cap_hands = cv2.VideoCapture('/Users/isaacroberts/Desktop/Engagement_Videos/' + level + '_Engagement/'+str(i)+'.m4v')
    video_to_extract = mp.VideoFileClip(r"/Users/isaacroberts/Desktop/Engagement_Videos/" + level + "_Engagement/"+str(i)+".m4v")
    audio = video_to_extract.audio.write_audiofile(
        r"/Users/isaacroberts/Desktop/Engagement_Videos/Audio_Data/" + level + "/"+str(i)+".wav")

    # Gaze and Blink

    gaze = GazeTracking()

    if not cap.isOpened():
        print("Incorrect file inputted!")

    while True:
        # We get a new frame from the webcam

        _, frame = cap.read()

        if _:
            # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)
            frame = gaze.annotated_frame()

            # use grayscale for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            current = ""

            b_count = 0
            r_count = 0
            l_count = 0
            c_count = 0
            d_count = 0

            if gaze.is_blinking():
                stage = "blinked"
                current = "Blinking"
                b_count += 1
            if not gaze.is_blinking() and stage == "blinked":
                stage = "done"
                blink_count += 1
            elif gaze.is_right():
                current = "Looking right"
                r_count += 1
            elif gaze.is_left():
                current = "Looking left"
                l_count += 1
            elif gaze.is_center():
                current = "Looking center"
                c_count += 1
            elif not gaze.is_blinking() and not gaze.is_right() and not gaze.is_left() and not gaze.is_center():
                current = "Looking down"
                d_count += 1

            if (r_count + l_count) > (b_count + c_count):
                gaze_direction_number = 3
            elif (d_count + b_count) > (r_count + l_count):
                gaze_direction_number = 2
            elif (b_count + c_count) > (r_count + l_count):
                gaze_direction_number = 1

            cv2.putText(frame, "Gaze: " + str(gaze_direction_number), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 0, 0), 2)
            cv2.putText(frame, f'Blink Count: {int(blink_count)}', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0),
                        2)
            #cv2.putText(frame, "Blinking: " + str(blink_count), (50, 530), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Gaze and Blink Detection", frame)

            # quit with q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture and video write objects
    cap.release()
    cv2.destroyAllWindows()

    # Pose

    if not cap_hands.isOpened():
        print("Incorrect file inputted!")

    while True:
        with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            _, frame = cap_hands.read()

            if _:

                # Flip the image horizontally for a later selfie-view display
                # Convert the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False

                # Process the image and find hands
                results = hands.process(image)

                image.flags.writeable = True

                # Draw the hand annotations on the image.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    stage = "pose"
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if not results.multi_hand_landmarks and stage == "pose":
                    stage = "done"
                    pose_count += 1

                # total / frames

                cv2.putText(image, f'Pose Count: {int(pose_count)}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0),
                            2)

                cv2.imshow('Pose Detection', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    cap_hands.release()
    cv2.destroyAllWindows()

    # Audio

    word_count = 0

    r = sr.Recognizer()

    with sr.AudioFile('/Users/isaacroberts/Desktop/Engagement_Videos/Audio_Data/' + level + '/'+str(i)+'.wav') as source:
        audio_text = r.listen(source)

        try:
            text = r.recognize_google(audio_text)
            words = text.split()
            word_count = len(words)

        except:
            print("Video has no sound")


    cv2.destroyAllWindows()

    # Emotion

    vid_title = i

    emotion_name = ""
    emotion_number = 0
    # angry = 1, disgust = 2, fear = 3, happy = 4
    # neutral = 5, sad = 6, surprise = 7
    print(i)

    try:
        face_detector = FER(mtcnn=True)
        input_video = Video(video_location)
        processing_data = input_video.analyze(face_detector, display=False)

        vid_df = input_video.to_pandas(processing_data)
        vid_df = input_video.get_first_face(vid_df)
        vid_df = input_video.get_emotions(vid_df)

        angry = sum(vid_df.angry)
        disgust = sum(vid_df.disgust)
        fear = sum(vid_df.fear)
        happy = sum(vid_df.happy)
        sad = sum(vid_df.sad)
        surprise = sum(vid_df.surprise)
        neutral = sum(vid_df.neutral)

        total = 0
        if angry > disgust and angry > fear and angry > happy and angry > sad and angry > surprise and angry > neutral:
            total = angry
            emotion_name = "Angry"
            emotion_number = 1
        elif happy > disgust and happy > fear and happy > angry and happy > sad and happy > surprise and happy > neutral:
            total = happy
            emotion_name = "Happy"
            emotion_number = 4
        elif disgust > happy and disgust > fear and disgust > angry and disgust > sad and disgust > surprise and disgust > neutral:
            total = disgust
            emotion_name = "Disgust"
            emotion_number = 2
        elif fear > happy and fear > disgust and fear > angry and fear > sad and fear > surprise and fear > neutral:
            total = fear
            emotion_name = "Fear"
            emotion_number = 3
        elif sad > happy and sad > disgust and sad > angry and sad > fear and sad > surprise and sad > neutral:
            total = sad
            emotion_name = "Sad"
            emotion_number = 6
        elif surprise > happy and surprise > disgust and surprise > angry and surprise > fear and surprise > sad and surprise > neutral:
            total = surprise
            emotion_name = "Surprise"
            emotion_number = 7
        elif neutral > happy and neutral > disgust and neutral > angry and neutral > fear and neutral > sad and neutral > surprise:
            total = neutral
            emotion_name = "Neutral"
            emotion_number = 5
    except:
        emotion_number = 0

    header = ['Video Engagement', 'Emotion', 'Gaze', 'Blink Count', 'Pose Count', 'Word Count']
    data = [" ", emotion_number, gaze_direction_number, blink_count, pose_count, word_count]

    with open('/Users/isaacroberts/Desktop/Engagement_Videos/Data/' + level + '/' + str(vid_title) + '.csv', 'w', encoding='UTF8',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)

    cv2.destroyAllWindows()

    i += 1

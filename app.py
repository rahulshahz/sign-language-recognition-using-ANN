from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask,render_template,Response,session
from PIL import ImageGrab
import cv2
import pickle
import mediapipe as mp
import numpy as np
from flask_socketio import SocketIO
import time
import pyscreenshot as ImageGrab
import schedule
from datetime import datetime
import sys
from tensorflow import keras
app=Flask(__name__)
socketio = SocketIO(app)

app.secret_key = 'your_secret_key'
camera=cv2.VideoCapture(0) # Replace with a strong secret key

# Dummy database (you should use a real database in production)
users = {
    'user1': {
        'username': 'user1',
        'password_hash': generate_password_hash('password1')
    }
}

model = keras.models.load_model("model.h5")

# Initialize the webcam
capture = cv2.VideoCapture(0)

# MediaPipe setup for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Sign language labels dictionary
labels_dict = {0: "A", 1: "I am", 2: "bad", 3: "good", 4: "hello", 5: "I love you", 6: "no", 7: "okay", 8: "why", 9: "yes",10:"you"}
reversed_labels_dict = {v: k for k, v in labels_dict.items()}
def generate_frames():
    lastElement = "string"
    # no_of_labels = 10
    # d = {}
    # for i in range(no_of_labels):
    #     d[i] = (0,0)
    # count = 0

    while True:
        # count += 1
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            data_aux = []
            x_ = []
            y_ = []
            # ret, frame = capture.read()
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10
                    data_aux.extend(data_aux)

                    data_aux = np.array(data_aux).reshape(1, 84)  # Reshape to (1, 84)
                    prediction = model.predict(data_aux)  # Use the reshaped data_aux

                    check = np.argmax(prediction[0])
                    print(check)
                    # sign_name = labels_dict[check]
                    
                    # data_aux = []
                    # x_ = []
                    # y_ = []
                    
                    # predicted_sign = labels_dict[check]
                    # print(check)
                    # d[check] = (d[check][0]+prediction_probability[0][check],d[check][1]+1)
                    # print(check,d[check])
                    
                    # prediction = model.predict([np.asarray(data_aux)])
                    # prediction probability
                    # prediction_probability = model.predict_proba([np.asarray(data_aux)])
                    # check = prediction[0]  
                    
                    predicted_sign = labels_dict[check]
                    
                    if lastElement != predicted_sign:
                        lastElement = predicted_sign
                        closedCaptions(predicted_sign)
                    data_aux = []
                    x_ = []
                    y_ = []

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def take_screenshot():
    image_name = f"screenshot-{str(datetime.now())}"
    image_name = image_name.replace(".","-")
    image_name = image_name.replace(":","-")
    image_name = image_name.replace(",","-")
    screenshot = ImageGrab.grab(bbox=(500,300,1500,900))
    filepath = f"./screenshots/{image_name}.jpg"
    screenshot.save(filepath)
    return filepath

def generate_frames_video_call():
    lastElement = "string"
        
    while True:
        # Create a video capture object for the screen
        time.sleep(0.2)
        screen_capture = take_screenshot()

        data_aux = []
        x_ = []
        y_ = []
        
        frame = cv2.cvtColor(cv2.imread(screen_capture), cv2.COLOR_RGB2BGR)
        H, W, _ = frame.shape
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                prediction = model.predict([np.asarray(data_aux)])
                
                predicted_sign = labels_dict[int(prediction[0])]
                
                if lastElement != predicted_sign:
                    lastElement = predicted_sign
                    print(predicted_sign)
                    closedCaptions(predicted_sign)
                data_aux = []
                x_ = []
                y_ = []

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')





@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videoCall')
def videoCall():
    return Response(generate_frames_video_call(),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('closedCaptions')
def closedCaptions(caption):
    socketio.emit('closedCaptions', caption)

@socketio.on('predictionProbability')
def predictionProbability(prompt):
    socketio.emit('predictionProbability', prompt)

@app.route('/tutorial')
def tutorial():
    if 'username' in session:
        return render_template("tutorial.html")
        # return f'Hello, {session["username"]}! <a href="/logout">Logout</a>'
    return render_template("index.html")

@app.route('/translator')
def meeting():
    if 'username' in session:
        return render_template("translator.html")
        # return f'Hello, {session["username"]}! <a href="/logout">Logout</a>'
    return render_template("index.html")

@app.route('/')
def home():
    if 'username' in session:
        return render_template("translator.html")
        # return f'Hello, {session["username"]}! <a href="/logout">Logout</a>'
    return render_template("index.html")



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return 'Username already exists. <a href="/signup">Try again</a>'
        users[username] = {
            'username': username,
            'password_hash': generate_password_hash(password)
        }
        session['username'] = username
        return redirect('/')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and check_password_hash(user['password_hash'], password):
            session['username'] = username
            return redirect('/')
        return 'Invalid login. <a href="/login">Try again</a>'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

if __name__ == '__main__':
    socketio.run(app, debug=True)
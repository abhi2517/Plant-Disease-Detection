from flask import Flask, render_template, request, session, redirect, flash, url_for, Response
from googletrans import Translator
import psycopg2
import sqlite3
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from utils import visualize
import time

app = Flask(__name__)
app.secret_key = "plant"
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = None
cap = None
COUNTER, FPS = 0, 0
START_TIME = time.time()
detection_result_list = []


def connect_to_db():
    conn = sqlite3.connect("plant.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enables fetching rows as dictionaries
    return conn
conn = connect_to_db()
cursor = conn.cursor()

# Create the `login` table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS login (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
""")
conn.commit()

# Routes
@app.route("/")
def login():
    return render_template("login.html")

@app.route('/add_users', methods=['POST'])
def add_users():
    name = request.form.get('uname')
    email = request.form.get('uemail')
    password = request.form.get('upassword')

    try:
        cursor.execute("""
            INSERT INTO login (name, email, password)
            VALUES (?, ?, ?)
        """, (name, email, password))
        conn.commit()
        return render_template("successfull.html")
    except sqlite3.IntegrityError:
        flash('Email already exists.', 'danger')
        return redirect('/')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    cursor.execute("SELECT user_id, name, email FROM login WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()

    if user:
        session['user_id'] = user['user_id']
        session['user_name'] = user['name']
        session['user_email'] = user['email']
        return redirect('/starter')
    else:
        flash('Invalid email or password', 'danger')
        return redirect('/')

@app.route('/starter')
def starter():
    name = session.get('user_name')
    if name:
        return render_template("new.html", name=name)
    else:
        flash('Please log in first.', 'warning')
        return redirect('/')


@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Process the image
            result_path, detected_disease = run_image_detection(file_path)

            # Render the result
            return render_template("result.html", result_image=result_path, detected_disease=detected_disease)

    return render_template("upload_image.html")

@app.route("/diagnosis/<disease>")
def diagnosis(disease):
    remedies_dict = {
        "Blight": "Remove infected plants, avoid overhead watering, and use fungicides like Chlorothalonil or Mancozeb.",
        "Rot": "Ensure proper drainage, avoid overwatering, and apply copper-based fungicides.",
        "Scab": "Use resistant varieties, maintain sanitation, and apply fungicides like Captan.",
        "Spot": "Prune infected areas, ensure air circulation, and use fungicides like Copper Sulfate.",
        "Powdery_Mildew": "Reduce humidity, increase air circulation, and use sulfur-based fungicides.",
        "Rust": "Remove infected leaves, avoid overhead watering, and use fungicides like Myclobutanil.",
        "Mold": "Maintain low humidity, improve air circulation, and use fungicides like Neem Oil."
    }
    remedies = remedies_dict.get(disease, "No remedies found for this disease.")
    return render_template("diagnosis.html", disease_name=disease, remedies=remedies)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text', '')
    target_language = data.get('target_language', '')

    translator = Translator()
    try:
        translation = translator.translate(text, dest=target_language)
        return {'translated_text': translation.text}
    except Exception as e:
        return {'error': str(e)}, 500

def run_image_detection(image_path: str) -> (str, str):
    """Run inference on an uploaded image and return the result image path and detected disease."""
    model_path = "plant.tflite"
    max_results = 5
    score_threshold = 0.7

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        flash(f"ERROR: Unable to read image at {image_path}. Check the file path.")
        return None, None

    # Resize and preprocess
    image = cv2.resize(image, (640, 480))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Initialize detector
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        max_results=max_results,
        score_threshold=score_threshold,
    )
    detector = vision.ObjectDetector.create_from_options(options)
    detection_result = detector.detect(mp_image)

    # Process results
    detected_disease = (
        detection_result.detections[0].categories[0].category_name
        if detection_result.detections else "None"
    )

    # Visualize results
    result_image = visualize(image, detection_result)
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
    cv2.imwrite(result_path, result_image)

    return result_path, detected_disease

def initialize_detector(model_path, max_results=5, score_threshold=0.8):
    """Initialize the Mediapipe object detector."""
    global detector
    try:
        print("Initializing detector...")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=max_results,
            score_threshold=score_threshold,
            result_callback=save_result,
        )
        detector = vision.ObjectDetector.create_from_options(options)
        print("Detector initialized successfully.")
    except Exception as e:
        print(f"ERROR: Detector initialization failed. {str(e)}")
        detector = None


def save_result(result, unused_output_image, timestamp_ms):
    """Callback to save detection results and calculate FPS."""
    global FPS, COUNTER, START_TIME, detection_result_list

    # Calculate FPS
    if COUNTER % 10 == 0:
        FPS = 10 / (time.time() - START_TIME)
        START_TIME = time.time()

    detection_result_list.append(result)
    COUNTER += 1

def generate_frames():
    """Generate video frames with real-time detection."""
    global cap, detector, detection_result_list

    if detector is None:
        raise RuntimeError("ERROR: Detector is not initialized. Ensure initialize_detector() is called.")

    while True:
        if not cap.isOpened():
            print("ERROR: Camera is not opened.")
            break

        success, frame = cap.read()
        if not success:
            print("ERROR: Failed to read frame from camera.")
            break

        # Flip and preprocess frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform detection
        try:
            detector.detect_async(mp_frame, time.time_ns() // 1_000_000)
        except Exception as e:
            print(f"Detection Error: {str(e)}")
            continue

        # Overlay FPS and detections
        cv2.putText(frame, f"FPS: {FPS:.1f}", (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if detection_result_list:
            frame = visualize(frame, detection_result_list[0])
            detection_result_list.clear()

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("ERROR: Failed to encode frame.")
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/start_detection')
def start_detection():
    """Start the camera and detection."""
    global cap, detector

    model_path = "plant.tflite"  # Path to the model
    if not os.path.exists(model_path):
        flash(f"ERROR: Model file '{model_path}' not found.")
        return redirect(url_for('starter'))

    # Initialize detector
    if detector is None:
        initialize_detector(model_path)
    if detector is None:
        flash("ERROR: Detector initialization failed.")
        return redirect(url_for('starter'))

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        flash("ERROR: Unable to access the camera.")
        return redirect(url_for('starter'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera started successfully.")

    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    """Route to stream video frames with detection."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_detection')
def stop_detection():
    """Release the camera and clean up."""
    global cap, detector

    # Release the camera if it's open
    if cap and cap.isOpened():
        cap.release()

    # Safely close the detector if it's initialized
    if detector:
        try:
            detector.close()
        except ValueError:
            pass  # Detector is already closed or not running

    return redirect(url_for('starter'))



@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import secrets
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
import os
import json
import threading
import time
import logging
import re
from dotenv import load_dotenv
import mediapipe as mp
# MediaPipe legacy solutions are failing in this environment. 
# We'll use OpenCV for face detection as it's more robust.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

load_dotenv()
print("DB_URI:", os.getenv("DB_URI"))
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.permanent_session_lifetime = timedelta(days=1)  # Set for "Remember Me"

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DB_URI")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('static'):
    os.makedirs('static')

# User Model
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(1023), nullable=False)
    profile_pic = db.Column(db.String(120), default='defaultprofile.jpg')

class SessionHistory(db.Model):
    __tablename__ = 'session_history'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    login_time = db.Column(db.DateTime, default=db.func.now())
    logout_time = db.Column(db.DateTime, nullable=True)
    remember_me = db.Column(db.Boolean, default=False)
    session_token = db.Column(db.String(64), unique=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    user = db.relationship('User', backref=db.backref('sessions', lazy=True))

class EmotionAnalysis(db.Model):
    __tablename__ = 'emotion_analysis'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    emotion = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    user = db.relationship('User', backref=db.backref('emotions', lazy=True))

# Emotion labels — must match training order (FER-2013 order)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Minimum confidence threshold — predictions below this are flagged as "uncertain"
CONFIDENCE_THRESHOLD = 0.25

# Model input size — FER-2013 standard
MODEL_INPUT_SIZE = (48, 48)

# Load the model dynamically — supports both .keras and .h5 formats
model = None
for model_path in ["emotion_model.keras", "emotion_model.h5"]:
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"[ML] Model loaded from '{model_path}'.")
            print(f"[ML] Input shape expected: {model.input_shape}")
            # Warm up the model to pre-compile its graph (speeds up first real inference)
            dummy = np.zeros((1, *MODEL_INPUT_SIZE, 1), dtype=np.float32)
            model.predict(dummy, verbose=0)
            print("[ML] Model warm-up complete.")
        except Exception as e:
            print(f"[ML] Error loading model from '{model_path}': {e}")
            model = None
        break
if model is None:
    print("[ML] WARNING: No model file found (emotion_model.keras / emotion_model.h5).")
    print("[ML] Run 'python train_model.py' to train a new model.")

# Global lock for MediaPipe (thread safety)
mp_lock = threading.Lock()

# --- Face Detection (OpenCV) -----------------------------------------------------------

def detect_faces(image):
    """Detect faces and return bounding boxes using OpenCV."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces_detected = face_cascade.detectMultiScale(grey, 1.3, 5)
    
    faces = []
    for (x, y, w_box, h_box) in faces_detected:
        faces.append((x, y, x + w_box, y + h_box))
    return faces


def detect_landmarks(image):
    """
    Fallback for landmarks using face detection. 
    Since mediapipe.solutions is missing, we just check for face presence.
    """
    faces = detect_faces(image)
    return faces if faces else None


# --- Core Prediction Pipeline -----------------------------------------------------------

def _preprocess_face(face_img):
    """
    Convert a BGR face crop to the exact format used during FER-2013 training:
    - Convert to greyscale (FER images are single-channel)
    - Apply CLAHE for adaptive contrast enhancement (helps in different lighting)
    - Resize to MODEL_INPUT_SIZE
    - Normalise pixel values to [0, 1]
    - Return shape (1, H, W, 1) ready for model.predict()
    """
    grey = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grey = clahe.apply(grey)
    resized = cv2.resize(grey, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    normalised = resized.astype(np.float32) / 255.0
    return normalised.reshape(1, *MODEL_INPUT_SIZE, 1)


def _predict_single(face_array):
    """Run inference and return raw probability array."""
    return model.predict(face_array, verbose=0)[0]


def predict_emotion(image):
    """
    High-accuracy emotion prediction pipeline:
    1. Detect face bounding box and crop to face region
    2. Greyscale + CLAHE preprocessing (matches FER-2013 training)
    3. Test-Time Augmentation: average predictions from original + horizontal flip
    4. Confidence thresholding: mark low-confidence predictions as 'uncertain'
    5. Return full probability dict for all 7 emotions
    """
    if model is None:
        return {'error': 'Model not loaded. Run train_model.py to create a model.'}

    try:
        h, w = image.shape[:2]

        # Step 1: Detect and crop to face region
        faces = detect_faces(image)
        if faces:
            x1, y1, x2, y2 = faces[0]  # Use the first detected face
            # Add a small padding (10%) around the bounding box for context
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.10)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            face_crop = image[y1:y2, x1:x2]
        else:
            # Fallback: use full image (e.g. for close-up photos)
            face_crop = image

        # Step 2: Preprocess
        processed = _preprocess_face(face_crop)

        # Step 3: Test-Time Augmentation (TTA) — horizontal flip
        # The face mesh is roughly symmetric, so averaging orig + flipped
        # stabilises predictions significantly
        flipped = np.flip(face_crop, axis=1).copy()  # horizontal mirror
        processed_flip = _preprocess_face(flipped)

        pred_orig = _predict_single(processed)
        pred_flip = _predict_single(processed_flip)

        # Average the two predictions
        prediction = (pred_orig + pred_flip) / 2.0

        # Step 4: Build result dict
        scores = {EMOTION_LABELS[i]: float(prediction[i]) for i in range(len(EMOTION_LABELS))}

        # Step 5: Confidence thresholding
        dominant_score = max(scores.values())
        if dominant_score < CONFIDENCE_THRESHOLD:
            # Mark as uncertain — the calling code should handle this gracefully
            scores['_uncertain'] = True

        return scores

    except Exception as exc:
        logging.error(f"[predict_emotion] Exception: {exc}", exc_info=True)
        return {'error': str(exc)}

class EmotionTracker:
    """
    Thread-safe tracker that aggregates emotion predictions across video frames.
    Uses Exponential Moving Average (EMA) weighting so that more recent frames
    contribute more to the dominant emotion result, reducing visual jitter.
    """
    WINDOW_SIZE = 10        # Keep last N frames in memory
    EMA_ALPHA   = 0.6       # Weight for the most recent frame ( >0.5 = recency bias )

    def __init__(self):
        self.emotion_scores = []   # List of (scores_dict, timestamp)
        self.ema_scores = {}       # Running EMA
        self.lock = threading.Lock()

    def add_emotion(self, scores):
        # Strip internal keys before storing
        clean = {k: v for k, v in scores.items() if not k.startswith('_')}
        with self.lock:
            self.emotion_scores.append((clean, datetime.now()))
            # Keep a sliding window
            if len(self.emotion_scores) > self.WINDOW_SIZE:
                self.emotion_scores.pop(0)
            # Update EMA
            if not self.ema_scores:
                self.ema_scores = dict(clean)
            else:
                for emotion in EMOTION_LABELS:
                    new_val = clean.get(emotion, 0.0)
                    prev    = self.ema_scores.get(emotion, 0.0)
                    self.ema_scores[emotion] = self.EMA_ALPHA * new_val + (1 - self.EMA_ALPHA) * prev

    def get_results(self):
        with self.lock:
            if not self.emotion_scores:
                return "neutral", {"neutral": 100.0}
            # Prefer EMA scores — they are temporally smoothed
            if self.ema_scores:
                total = sum(self.ema_scores.values())
                if total > 0:
                    pct = {e: (v / total) * 100 for e, v in self.ema_scores.items()}
                    dominant = max(pct, key=pct.get)
                    return dominant, pct
            # Fallback: simple accumulation over window
            emotion_totals = {}
            for s, _ in self.emotion_scores:
                for emotion, score in s.items():
                    if score > 0:
                        emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score
            total = sum(emotion_totals.values())
            if total == 0:
                return "neutral", {"neutral": 100.0}
            pct = {e: (v / total) * 100 for e, v in emotion_totals.items()}
            return max(pct, key=pct.get), pct

    def clear(self):
        with self.lock:
            self.emotion_scores = []
            self.ema_scores = {}

emotion_trackers = {}

# Session validation middleware
@app.before_request
def validate_session():
    # Skip validation for static files and non-auth routes
    if request.endpoint in ['login', 'register', 'static', 'contact', 'about']:
        return
        
    if 'session_token' in session and 'user_id' in session:
        session_record = SessionHistory.query.filter_by(
            user_id=session['user_id'], 
            session_token=session['session_token'],
            logout_time=None
        ).first()
        if not session_record:
            session.clear()
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
                return jsonify({'error': 'Your session is invalid. Please log in again.'}), 401
            flash('Your session is invalid. Please log in again.', 'warning')
            return redirect(url_for('login'))
        if session_record.expires_at and datetime.now() > session_record.expires_at:
            session_record.logout_time = datetime.now()
            db.session.commit()
            session.clear()
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
                return jsonify({'error': 'Your session has expired. Please log in again.'}), 401
            flash('Your session has expired. Please log in again.', 'warning')
            return redirect(url_for('login'))

@app.route('/')
def default():
    return redirect(url_for('about'))

@app.route('/about')
def about():
    if 'user_id' not in session:
        return render_template('about.html')
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember_me = 'remember_me' in request.form
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            # Clear any existing sessions for this user
            SessionHistory.query.filter_by(user_id=user.id, logout_time=None).update({'logout_time': datetime.now()})
            db.session.commit()

            session['user_id'] = user.id
            session['username'] = user.username
            session['session_token'] = secrets.token_hex(32)
            session.permanent = remember_me
            expires_at = datetime.now() + timedelta(days=1) if remember_me else None
            new_session = SessionHistory(
                user_id=user.id,
                remember_me=remember_me,
                session_token=session['session_token'],
                expires_at=expires_at
            )
            db.session.add(new_session)
            db.session.commit()
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        existing_user = User.query.filter((User.email == email) | (User.username == username)).first()
        if existing_user:
            flash('Email or Username already exists', 'danger')
            return redirect(url_for('register'))
        new_user = User(username=username, email=email, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/logout')
def logout():
    if 'user_id' in session:
        user_id = session['user_id']
        session_token = session.get('session_token')
        session_record = SessionHistory.query.filter_by(
            user_id=user_id, 
            session_token=session_token,
            logout_time=None
        ).first()
        if session_record:
            session_record.logout_time = datetime.now()
            db.session.commit()
        session.clear()
        flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/cleanup_sessions')
def cleanup_sessions():
    expired_sessions = SessionHistory.query.filter(
        SessionHistory.logout_time.is_(None),
        SessionHistory.expires_at.isnot(None),
        SessionHistory.expires_at < datetime.now()
    ).all()
    count = 0
    for session_record in expired_sessions:
        session_record.logout_time = datetime.now()
        count += 1
    db.session.commit()
    return jsonify({'success': True, 'message': f'Cleaned up {count} expired sessions'})

@app.route('/invalidate_session', methods=['POST'])
def invalidate_session():
    if 'user_id' in session and 'session_token' in session:
        session_record = SessionHistory.query.filter_by(
            user_id=session['user_id'],
            session_token=session['session_token'],
            logout_time=None
        ).first()
        if session_record and not session_record.remember_me:
            session_record.logout_time = datetime.now()
            db.session.commit()
            session.clear()
            return jsonify({'success': True, 'message': 'Session invalidated'})
    return jsonify({'success': False, 'message': 'No active session'})

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    if 'user_id' in session and 'session_token' in session:
        session_record = SessionHistory.query.filter_by(
            user_id=session['user_id'],
            session_token=session['session_token'],
            logout_time=None
        ).first()
        if session_record and not session_record.remember_me:
            return jsonify({'success': True})
    session.clear()
    return jsonify({'success': False, 'message': 'Session expired'})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    file = request.files['image']
    if not file:
        return redirect(url_for('camera'))
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    landmarks = detect_landmarks(image)
    if landmarks is None:
        return redirect(url_for('camera'))
    
    scores = predict_emotion(image)
    if 'error' in scores:
        flash(scores['error'], 'danger')
        return redirect(url_for('camera'))
        
    dominant_emotion = max(scores, key=scores.get)
    confidence = float(scores[dominant_emotion])
    
    new_analysis = EmotionAnalysis(
        user_id=session['user_id'],
        emotion=dominant_emotion,
        confidence=confidence
    )
    db.session.add(new_analysis)
    db.session.commit()
    
    return render_template('analysis.html', emotion=scores, landmarks=landmarks.tolist())

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    user_id = session['user_id']
    session_records = SessionHistory.query.filter_by(user_id=user_id).order_by(SessionHistory.login_time.desc()).all()
    sessions_with_emotions = []
    for session_record in session_records:
        emotions_in_session = EmotionAnalysis.query.filter_by(user_id=user_id).filter(
            EmotionAnalysis.timestamp >= session_record.login_time,
            EmotionAnalysis.timestamp <= (session_record.logout_time if session_record.logout_time else datetime.now())
        ).all()
        predominant_emotion = "None" if not emotions_in_session else max((e.emotion for e in emotions_in_session),
                                                                       key=lambda x: sum(1 for e in emotions_in_session if e.emotion == x))
        sessions_with_emotions.append({
            'session': session_record,
            'predominant_emotion': predominant_emotion,
            'emotion_count': len(emotions_in_session)
        })
    emotion_records = EmotionAnalysis.query.filter_by(user_id=user_id).order_by(EmotionAnalysis.timestamp.desc()).all()
    return render_template('history.html',
                          sessions_with_emotions=sessions_with_emotions,
                          emotion_records=emotion_records)

@app.route('/session_emotions/<int:session_id>')
def session_emotions(session_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    user_id = session['user_id']
    session_record = SessionHistory.query.filter_by(id=session_id, user_id=user_id).first()
    if not session_record:
        return jsonify({'error': 'Session not found'}), 404
    emotions_in_session = EmotionAnalysis.query.filter_by(user_id=user_id).filter(
        EmotionAnalysis.timestamp >= session_record.login_time,
        EmotionAnalysis.timestamp <= session_record.logout_time if session_record.logout_time else datetime.now()
    ).all()
    emotion_counts = {}
    for emotion_record in emotions_in_session:
        emotion_counts[emotion_record.emotion] = emotion_counts.get(emotion_record.emotion, 0) + 1
    total = len(emotions_in_session) if emotions_in_session else 1
    emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
    return jsonify({
        'session_id': session_id,
        'emotions': emotion_counts,
        'percentages': emotion_percentages,
        'total_records': total
    })

@app.route('/get_profile', methods=['GET'])
def get_profile():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    first_login = SessionHistory.query.filter_by(user_id=user.id).order_by(SessionHistory.login_time.asc()).first()
    days_active = (datetime.now() - first_login.login_time).days if first_login else 0
    total_activities = SessionHistory.query.filter_by(user_id=user.id).count()
    return jsonify({
        'success': True,
        'username': user.username,
        'email': user.email,
        'profile_pic': url_for('static', filename=user.profile_pic),
        'days_active': days_active,
        'total_activities': total_activities
    })

@app.route('/show_profile', methods=['GET'])
def show_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html')

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404

    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    existing_user = User.query.filter(
        ((User.username == username) | (User.email == email)) & (User.id != user.id)
    ).first()
    if existing_user:
        return jsonify({'success': False, 'message': 'Username or email already exists'}), 400

    if username:
        user.username = username
    if email:
        user.email = email
    if password:
        user.password = generate_password_hash(password)

    if 'profile-pic-upload' in request.files:
        file = request.files['profile-pic-upload']
        if file and file.filename:
            from werkzeug.utils import secure_filename
            existing_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(f"{user.username}_pic")]
            max_version = 0
            for f in existing_files:
                try:
                    version = int(f.split('_pic')[-1].split('.')[0])
                    max_version = max(max_version, version)
                except ValueError:
                    continue
            new_version = max_version + 1
            filename = f"{user.username}_pic{new_version}.jpg"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            user.profile_pic = f"uploads/{filename}"

    db.session.commit()
    return jsonify({
        'success': True,
        'username': user.username,
        'email': user.email,
        'profile_pic': url_for('static', filename=user.profile_pic)
    })

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Extract form data (optional)
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Flash a success message
        flash('Thank you for your message! We will get back to you soon.', 'success')
        
        # Redirect to the same page to avoid form resubmission on refresh
        return redirect(url_for('contact'))
        
    return render_template('contact.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'})
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image file'})

    if model is None:
        return jsonify({'error': '⚠️ Emotion model not loaded. Please run train_model.py to create emotion_model.keras first.'})

    landmarks = detect_landmarks(image)
    if landmarks is None:
        return jsonify({'error': '😶 No face detected. Ensure your face is well-lit and facing the camera.'})

    scores = predict_emotion(image)
    if 'error' in scores:
        return jsonify({'error': scores['error']})

    # Remove internal helper keys before returning
    is_uncertain = scores.pop('_uncertain', False)
    if is_uncertain:
        return jsonify({'error': '🤔 Confidence too low. Improve lighting or move closer to the camera.'})

    dominant_emotion = max(scores, key=scores.get)
    confidence = float(scores[dominant_emotion])

    try:
        new_analysis = EmotionAnalysis(
            user_id=session['user_id'],
            emotion=dominant_emotion,
            confidence=confidence
        )
        db.session.add(new_analysis)
        db.session.commit()
    except Exception as e:
        logging.warning(f"DB save failed (non-fatal): {e}")

    return jsonify({
        'emotion': dominant_emotion,
        'scores': json.dumps(scores)
    })

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/videoscan')
def videoscan():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('videoscan.html')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/process_upload', methods=['POST'])
def process_upload():
    if 'user_id' not in session:
        logger.debug("User not logged in, redirecting to login")
        return redirect(url_for('login'))
    if 'image' not in request.files:
        logger.debug("No file part in request")
        flash('No file part', 'danger')
        return redirect(url_for('upload'))
    file = request.files['image']
    if file.filename == '':
        logger.debug("No file selected")
        flash('No selected file', 'danger')
        return redirect(url_for('upload'))
    logger.debug(f"Processing file: {file.filename}")
    # Check if model is loaded before doing any work
    if model is None:
        flash('⚠️ Emotion model is not loaded. Please run python train_model.py to create the model first.', 'danger')
        return redirect(url_for('upload'))
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.debug("Failed to read image")
            flash('Failed to read image. Please upload a valid image file.', 'danger')
            return redirect(url_for('upload'))
        logger.debug("Image read successfully")
        landmarks = detect_landmarks(image)
        if landmarks is None:
            logger.debug("No face detected in image")
            flash('No face detected in the image', 'warning')
            return redirect(url_for('upload'))
        logger.debug("Facial landmarks detected")
        scores = predict_emotion(image)
        if 'error' in scores:
            logger.debug(f"Emotion prediction error: {scores['error']}")
            flash(f'Error in emotion prediction: {scores["error"]}', 'danger')
            return redirect(url_for('upload'))
        logger.debug(f"Emotion scores: {scores}")
        dominant_emotion = max(scores, key=scores.get)
        confidence = float(scores[dominant_emotion])
        logger.debug(f"Dominant emotion: {dominant_emotion}")
        
        new_analysis = EmotionAnalysis(
            user_id=session['user_id'],
            emotion=dominant_emotion,
            confidence=confidence
        )
        db.session.add(new_analysis)
        db.session.commit()
        logger.debug("Analysis saved to database")
        return redirect(url_for('result', emotion=dominant_emotion, scores=json.dumps(scores)))
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('upload'))

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    if user_id not in emotion_trackers:
        emotion_trackers[user_id] = EmotionTracker()
    
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        last_emotion_time = time.time()
        emotion_interval = 1.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = time.time()
                dominant_emotion = 'neutral'  # safe default
                confidence = 0.0
                if current_time - last_emotion_time > emotion_interval:
                    emotion_scores = predict_emotion(frame)
                    if 'error' not in emotion_scores:
                        emotion_trackers[user_id].add_emotion(emotion_scores)
                        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                        confidence = emotion_scores[dominant_emotion]
                        if len(emotion_trackers[user_id].emotion_scores) % 5 == 0:
                            with app.app_context():
                                new_analysis = EmotionAnalysis(
                                    user_id=user_id,
                                    emotion=dominant_emotion,
                                    confidence=confidence
                                )
                                db.session.add(new_analysis)
                                db.session.commit()
                        last_emotion_time = current_time
                    cv2.putText(
                        frame,
                        f"Emotion: {dominant_emotion} ({confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                faces = detect_faces(frame)
                for (x1, y1, x2, y2) in faces:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in video stream: {e}")
        finally:
            cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        img_data = data.get('image', '')
        img_data = re.sub('^data:image/.+;base64,', '', img_data)
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image data'})
        
        landmarks = detect_landmarks(image)
        if landmarks is None:
            return jsonify({'success': False, 'error': 'No face detected'})
        
        scores = predict_emotion(image)
        if 'error' in scores:
            return jsonify({'success': False, 'error': scores['error']})
        
        dominant_emotion = max(scores, key=scores.get)
        confidence = float(scores[dominant_emotion])
        
        faces = detect_faces(image)
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Emotion: {dominant_emotion} ({confidence:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', image)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        user_id = session['user_id']
        if user_id not in emotion_trackers:
            emotion_trackers[user_id] = EmotionTracker()
        emotion_trackers[user_id].add_emotion(scores)
        
        try:
            new_analysis = EmotionAnalysis(
                user_id=user_id,
                emotion=dominant_emotion,
                confidence=confidence
            )
            db.session.add(new_analysis)
            db.session.commit()
        except Exception as db_error:
            print(f"Database error: {db_error}")
        
        return jsonify({
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{processed_image}',
            'emotion': dominant_emotion,
            'confidence': confidence,
            'scores': scores
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/end_session')
def end_session():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    if user_id in emotion_trackers:
        dominant_emotion, scores = emotion_trackers[user_id].get_results()
        emotion_trackers[user_id].clear()
        suggestions_dict = {
            'happy': ["Keep spreading positivity—share your joy with others! <a href='https://www.youtube.com/watch?v=X1GNc70-584' target='_blank'>Watch this podcast</a>", "Try a new hobby to maintain your high spirits.", "Reflect on what’s making you happy and how to sustain it."],
            'sad': ["Take a moment to relax—maybe watch a comforting movie. <a href='https://www.youtube.com/watch?v=h-3bixYKBFg' target='_blank'>Watch this podcast</a>", "Talk to a friend or loved one for support.", "Consider journaling your feelings to process them."],
            'angry': ["Take deep breaths or step away to cool off. <a href='https://www.youtube.com/watch?v=J8LMUkuxkbU' target='_blank'>Watch this podcast</a>", "Engage in physical activity like a quick walk to release tension.", "Write down what’s bothering you to gain perspective."],
            'fear': ["Focus on what you can control to ease your worries. <a href='https://www.youtube.com/watch?v=q6VRPyX1qHg' target='_blank'>Watch this podcast</a>", "Practice a grounding exercise, like counting to 10 slowly.", "Talk to someone you trust about what’s on your mind."],
            'disgust': ["Shift your focus to something you enjoy to reset. <a href='https://www.youtube.com/watch?v=oUTEJGEGGPE' target='_blank'>Watch this podcast</a>", "Take a break from whatever’s bothering you.", "Clean or organize your space for a fresh start."],
            'neutral': ["Try something new to spark some excitement. <a href='https://www.youtube.com/watch?v=x4JzmIZAPxs' target='_blank'>Watch this podcast</a>", "Set a small goal for the day to feel accomplished. <a href='https://www.youtube.com/watch?v=tnVMB0P-FYM' target='_blank'>Watch this podcast</a>", "Take a moment to appreciate the calm. <a href='https://www.youtube.com/watch?v=QnEVmSFJB9g' target='_blank'>Watch this podcast</a>"],
            'surprise': ["Embrace the unexpected—see where it takes you! <a href='https://www.youtube.com/watch?v=2kdvTv7jg1s' target='_blank'>Watch this podcast</a>", "Pause and assess what surprised you to stay grounded.", "Share your reaction with someone for a fun conversation."]
        }
        suggestions = suggestions_dict.get(dominant_emotion.lower(), ["Take a moment to reflect on your current state."])
        return render_template('result.html',
                             emotion=dominant_emotion,
                             scores=scores,
                             pie_chart_data=pie_chart_base64,
                             suggestions=suggestions)
    flash('No emotion data recorded', 'warning')
    return redirect(url_for('videoscan'))

@app.route('/current_emotion_stats')
def current_emotion_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    user_id = session['user_id']
    if user_id in emotion_trackers:
        dominant_emotion, scores = emotion_trackers[user_id].get_results()
        return jsonify({
            'dominant_emotion': dominant_emotion,
            'scores': scores
        })
    return jsonify({'error': 'No emotion data available'}), 404

@app.route('/result')
def result():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    emotion = request.args.get('emotion', 'Unknown')
    scores_str = request.args.get('scores', '{}')
    try:
        scores = json.loads(scores_str.replace("'", '"'))
    except json.JSONDecodeError:
        scores = {}
    suggestions_dict = {
        'happy': ["Keep spreading positivity—share your joy with others! <a href='https://www.youtube.com/watch?v=X1GNc70-584' target='_blank'>Watch this podcast</a>", "Try a new hobby to maintain your high spirits.", "Reflect on what's making you happy and how to sustain it."],
        'sad': ["Take a moment to relax—maybe watch a comforting movie. <a href='https://www.youtube.com/watch?v=h-3bixYKBFg' target='_blank'>Watch this podcast</a>", "Talk to a friend or loved one for support.", "Consider journaling your feelings to process them."],
        'angry': ["Take deep breaths or step away to cool off. <a href='https://www.youtube.com/watch?v=J8LMUkuxkbU' target='_blank'>Watch this podcast</a>", "Engage in physical activity like a quick walk to release tension.", "Write down what's bothering you to gain perspective."],
        'fear': ["Focus on what you can control to ease your worries. <a href='https://www.youtube.com/watch?v=q6VRPyX1qHg' target='_blank'>Watch this podcast</a>", "Practice a grounding exercise, like counting to 10 slowly.", "Talk to someone you trust about what's on your mind."],
        'disgust': ["Shift your focus to something you enjoy to reset. <a href='https://www.youtube.com/watch?v=oUTEJGEGGPE' target='_blank'>Watch this podcast</a>", "Take a break from whatever's bothering you.", "Clean or organize your space for a fresh start."],
        'neutral': ["Try something new to spark some excitement. <a href='https://www.youtube.com/watch?v=x4JzmIZAPxs' target='_blank'>Watch this podcast</a>", "Set a small goal for the day to feel accomplished. <a href='https://www.youtube.com/watch?v=tnVMB0P-FYM' target='_blank'>Watch this podcast</a>", "Take a moment to appreciate the calm. <a href='https://www.youtube.com/watch?v=QnEVmSFJB9g' target='_blank'>Watch this podcast</a>"],
        'surprise': ["Embrace the unexpected—see where it takes you! <a href='https://www.youtube.com/watch?v=2kdvTv7jg1s' target='_blank'>Watch this podcast</a>", "Pause and assess what surprised you to stay grounded.", "Share your reaction with someone for a fun conversation."]
    }
    suggestions = suggestions_dict.get(emotion.lower(), ["Take a moment to reflect on your current state."])
    return render_template('result.html',
                         emotion=emotion,
                         scores=scores,
                         suggestions=suggestions)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
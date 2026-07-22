import os
import cv2
import json
import base64
import socket
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# Directory to save snapshots and global match data
SNAPSHOT_DIR = os.path.join(app.root_path, 'snapshots')
DATA_FILE = os.path.join(app.root_path, 'match_data.json')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Helper to get Local IP Address for Global Network Access
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# Initialize global match_data.json if not present
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump({'history': [], 'leaderboard': {}}, f, indent=2)

class AirWriterEngine:
    def __init__(self):
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.draw_color = (255, 0, 128)
        self.brush_thickness = 5
        self.hands = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        except Exception as e:
            print(f"MediaPipe notice: {e}")
            self.hands = None

    def process_frame(self, frame):
        if frame is None:
            return None

        h, w, c = frame.shape
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        frame = cv2.flip(frame, 1)

        if self.hands:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    lm = hand_landmarks.landmark
                    idx_x, idx_y = int(lm[8].x * w), int(lm[8].y * h)

                    idx_up = lm[8].y < lm[6].y
                    mid_up = lm[12].y < lm[10].y
                    ring_up = lm[16].y < lm[14].y
                    pinky_up = lm[20].y < lm[18].y

                    if idx_up and not mid_up and not ring_up and not pinky_up:
                        cv2.circle(frame, (idx_x, idx_y), self.brush_thickness + 2, self.draw_color, -1)
                        if self.prev_x == 0 and self.prev_y == 0:
                            self.prev_x, self.prev_y = idx_x, idx_y
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (idx_x, idx_y), self.draw_color, self.brush_thickness)
                        self.prev_x, self.prev_y = idx_x, idx_y

                    elif idx_up and mid_up and not ring_up and not pinky_up:
                        self.prev_x, self.prev_y = 0, 0
                        cv2.circle(frame, (idx_x, idx_y), 12, (0, 255, 255), 2)
                        cv2.putText(frame, "HOVER", (idx_x - 20, idx_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    elif idx_up and mid_up and ring_up and not pinky_up:
                        self.reset_canvas()
                        self.prev_x, self.prev_y = 0, 0
                        cv2.putText(frame, "CANVAS CLEARED", (idx_x - 40, idx_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        self.prev_x, self.prev_y = 0, 0
            else:
                self.prev_x, self.prev_y = 0, 0

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
        frame_bg = cv2.bitwise_and(frame, inv_canvas)
        output_frame = cv2.add(frame_bg, self.canvas)
        return output_frame

    def reset_canvas(self):
        if self.canvas is not None:
            self.canvas.fill(0)

engine = AirWriterEngine()

def generate_video_stream():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera not detected for server-side stream.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        processed_frame = engine.process_frame(frame)
        if processed_frame is None:
            processed_frame = frame

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/save_canvas', methods=['POST'])
def save_canvas():
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_writing_{timestamp}.png"
        filepath = os.path.join(SNAPSHOT_DIR, filename)

        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        return jsonify({'status': 'success', 'filename': filename, 'path': filepath})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'history': [], 'leaderboard': {}})

@app.route('/api/save_match', methods=['POST'])
def save_match():
    try:
        match_info = request.get_json()
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        data['history'].insert(0, match_info)
        data['history'] = data['history'][:100]

        p1 = match_info.get('player1', 'Player X')
        p2 = match_info.get('player2', 'Player O')
        winner = match_info.get('winner', 'Tie')

        lb = data.get('leaderboard', {})
        for p in [p1, p2]:
            if p not in lb:
                lb[p] = {'wins': 0, 'losses': 0, 'ties': 0, 'streak': 0, 'maxStreak': 0, 'total': 0}

        if winner == 'Tie':
            lb[p1]['ties'] += 1
            lb[p2]['ties'] += 1
            lb[p1]['streak'] = 0
            lb[p2]['streak'] = 0
        elif winner == p1:
            lb[p1]['wins'] += 1
            lb[p1]['streak'] += 1
            lb[p1]['maxStreak'] = max(lb[p1]['maxStreak'], lb[p1]['streak'])
            lb[p2]['losses'] += 1
            lb[p2]['streak'] = 0
        else:
            lb[p2]['wins'] += 1
            lb[p2]['streak'] += 1
            lb[p2]['maxStreak'] = max(lb[p2]['maxStreak'], lb[p2]['streak'])
            lb[p1]['losses'] += 1
            lb[p1]['streak'] = 0

        lb[p1]['total'] += 1
        lb[p2]['total'] += 1
        data['leaderboard'] = lb

        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        return jsonify({'status': 'success', 'history': data['history'], 'leaderboard': lb})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    try:
        empty_data = {'history': [], 'leaderboard': {}}
        with open(DATA_FILE, 'w') as f:
            json.dump(empty_data, f, indent=2)
        return jsonify({'status': 'success', 'history': [], 'leaderboard': {}})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    local_ip = get_local_ip()
    use_https = os.environ.get('USE_HTTPS', '1') == '1'

    print("============================================================")
    print("GLOBAL ACCESS FLASK SERVER RUNNING")
    if use_https:
        print(f"HTTPS Local Access:   https://127.0.0.1:5000")
        print(f"HTTPS Network Access: https://{local_ip}:5000")
        print("CAMERA ACCESS: ENABLED (HTTPS Secure Context)")
    else:
        print(f"HTTP Local Access:   http://127.0.0.1:5000")
        print(f"HTTP Network Access: http://{local_ip}:5000")
    print("============================================================")

    try:
        if use_https:
            app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
        else:
            app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as err:
        print(f"HTTPS Fallback to HTTP: {err}")
        app.run(host='0.0.0.0', port=5000, debug=True)

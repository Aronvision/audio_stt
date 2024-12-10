import os
from flask import Flask, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, 'static')

# Create static directory if it doesn't exist
os.makedirs(static_dir, exist_ok=True)

app = Flask(__name__, static_folder=static_dir)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(static_dir, filename)

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Live Camera Stream</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f9;
                    position: relative;
                    height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .container {
                    display: none;  /* Initially hidden */
                    position: relative;
                    width: 100%;
                    height: 100%;
                }
                .nukki-image {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    z-index: 10;
                    width: 400px;
                }
                .text-left {
                    position: absolute;
                    top: 50%;
                    left: 20%;
                    transform: translate(0, -50%);
                    font-size: 70px;
                    font-weight: bold;
                    color: #333;
                }
                .text-right {
                    position: absolute;
                    top: 50%;
                    right: 20%;
                    transform: translate(0, -50%);
                    font-size: 70px;
                    font-weight: bold;
                    color: #333;
                }
                .stop-button {
                    position: absolute;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    padding: 15px 30px;
                    font-size: 24px;
                    background-color: #ff4444;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                .stop-button:hover {
                    background-color: #cc0000;
                }
                .subtitle {
                    position: absolute;
                    bottom: 100px;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 24px;
                    color: #333;
                    background-color: rgba(255, 255, 255, 0.9);
                    padding: 15px 30px;
                    border-radius: 10px;
                    text-align: center;
                    max-width: 80%;
                    display: none;
                }
            </style>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    var socket = io();
                    var container = document.querySelector('.container');
                    var stopButton = document.querySelector('.stop-button');
                    var subtitle = document.querySelector('.subtitle');
                    
                    socket.on('connect', function() {
                        console.log('Connected to server');
                    });
                    
                    socket.on('stt_event', function(data) {
                        console.log('Received event:', data);
                        
                        if (!data || !data.status) {
                            console.error('Invalid event data received');
                            return;
                        }
                        
                        try {
                            switch(data.status) {
                                case 'start_stt':
                                    container.style.display = 'block';
                                    stopButton.style.display = 'block';
                                    stopButton.disabled = false;
                                    subtitle.style.display = 'none';
                                    document.querySelector('.text-left').textContent = '질문';
                                    document.querySelector('.text-right').textContent = '인식중!';
                                    break;
                                    
                                case 'stop_stt':
                                    container.style.display = 'none';
                                    stopButton.style.display = 'none';
                                    break;
                                    
                                case 'chatgpt_response':
                                    if (data.response) {
                                        subtitle.textContent = data.response;
                                        subtitle.style.display = 'block';
                                        document.querySelector('.text-left').textContent = '답변';
                                        document.querySelector('.text-right').textContent = '생성완료!';
                                        console.log('Updated subtitle with:', data.response);
                                    } else {
                                        console.error('Missing response in chatgpt_response event');
                                    }
                                    break;
                                    
                                default:
                                    console.warn('Unknown status:', data.status);
                            }
                        } catch (error) {
                            console.error('Error handling event:', error);
                        }
                    });

                    stopButton.addEventListener('click', function() {
                        socket.emit('stop_recording');
                        stopButton.disabled = true;
                        stopButton.style.display = 'none';
                    });
                });
            </script>
        </head>
        <body>
            <div class="container">
                <div class="text-left">질문</div>
                <div class="text-right">인식중!</div>
                <img src="/static/al_model.png" alt="Nukki Image" class="nukki-image">
                <button class="stop-button">음성 인식 종료</button>
                <div class="subtitle"></div>
            </div>
        </body>
        </html>
    ''')

@socketio.on('stt_status')
def handle_stt_status(data):
    """Handle STT status updates with improved validation"""
    print(f"Received STT status: {data}")
    
    try:
        # Validate incoming data
        if not isinstance(data, dict):
            print(f"Invalid data format received: {type(data)}")
            return
        
        # Ensure required fields are present
        if 'status' not in data:
            if isinstance(data, str):
                # Convert string status to proper format
                data = {'status': data}
            else:
                print("Missing 'status' field in data")
                return
        
        # Forward the event to all connected clients
        socketio.emit('stt_event', data)
        print(f"Successfully forwarded STT event: {data}")
        
    except Exception as e:
        print(f"Error handling STT status: {str(e)}")

@socketio.on('stop_recording')
def handle_stop_recording():
    print("Stop recording signal received from web interface")
    socketio.emit('stt_event', {'status': 'stop_recording'})

if __name__ == '__main__':
    print(f"Static folder path: {static_dir}")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)



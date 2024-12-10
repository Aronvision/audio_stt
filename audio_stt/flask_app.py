from flask import Flask, jsonify, render_template_string
import threading

app = Flask(__name__)

# STT 컨트롤 HTML 템플릿
STT_TEMPLATE = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>로봇 컨트롤 패널</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
            }
            .container {
                text-align: center;
                padding: 20px;
            }
            button {
                padding: 15px 30px;
                font-size: 18px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
        <script>
            function startSTT() {
                fetch('/start_stt')
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        if (data.status === "completed") {
                            window.location.href = '/';
                        }
                    })
                    .catch(error => alert('Error: ' + error));
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>음성 인식 제어</h1>
            <button onclick="startSTT()">음성 인식 시작</button>
        </div>
    </body>
    </html>
'''

@app.route('/')
def index():
    return render_template_string(STT_TEMPLATE)

@app.route('/start_stt')
def start_stt():
    # 서비스 호출 코드 제거됨
    return jsonify({"status": "error", "message": "Not implemented"}), 501

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

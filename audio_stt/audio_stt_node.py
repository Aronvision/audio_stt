import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32
from sensor_msgs.msg import Image as ROSImage  # ROS의 Image와 구분하기 위해 이름 변경
from cv_bridge import CvBridge
import ast
import openai
from .module import Audio_record, Custom_faster_whisper, Custom_TTS
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # PIL 모듈 추가
import threading
import queue
import time  # Added this import
import subprocess  # Add this import at the top with other imports
import simpleaudio as sa


class SttGui:
    def __init__(self):
        self.root = None
        self.current_mode = 1  # 현재 모드 추적
        self.stt_indicator = None
        self.subtitle_label = None

    def create_window(self):
        # 메인 윈도우 설정
        self.root = tk.Tk()
        self.root.protocol('WM_DELETE_WINDOW', self._on_closing)

        # 화면 크기 설정
        self.root.attributes('-fullscreen', True)

        # 배경색 지정
        self.root.configure(bg='#dce8d5')

        self.show_mode1()  # 초기 모드 표시

    def show_mode1(self):
        # 기존 위젯들 제거
        for widget in self.root.winfo_children():
            widget.destroy()

        top_spacer = tk.Frame(self.root, height=30, bg='#dce8d5')  # Reduced spacing
        top_spacer.pack()
        
        self.top_label = tk.Label(
            self.root, 
            text="한양병원에 오신 것을 환영합니다!", 
            font=("Helvetica", 50),  # Reduced font size from 70
            bg='#dce8d5',
            wraplength=1000,  # Reduced wraplength
            pady=30  # Reduced padding
        )

        self.top_label.pack(pady=(100))  # Reduced top padding

        # 캐릭터 이미지 표시
        image_path = os.path.join(os.path.dirname(__file__), "templete", "al_model.png")
        self.character_image = Image.open(image_path)
        self.character_image = self.character_image.resize((500, 750))  # Reduced image size
        self.character_photo = ImageTk.PhotoImage(self.character_image)
        self.character_label = tk.Label(
            self.root,
            image=self.character_photo,
            bg='#dce8d5'
        )
        self.character_label.pack(pady=(20, 30))  # Reduced padding

        self.current_mode = 1

    def show_mode2(self):
        # 기존 위젯들 제거
        for widget in self.root.winfo_children():
            widget.destroy()

        top_spacer = tk.Frame(self.root, height=50, bg='#dce8d5')
        top_spacer.pack()

        # 모드 2 UI 구성 - 배경색 변경
        self.top_label = tk.Label(
            self.root, 
            text="저는 병원 안내로봇 알쏭이입니다!",
            font=("Helvetica", 45),
            bg='#ADD8E6',
            wraplength=1000,
            pady=30
        )
        self.top_label.pack(pady=(30, 0))  # Reduced bottom padding

        # Add new label for the red text
        self.red_label = tk.Label(
            self.root,
            text="※음성안내를 따라주세요※",
            font=("Helvetica", 25),
            bg='#ADD8E6',
            fg='red',  # Text color set to red
            wraplength=1000,
            pady=20
        )
        self.red_label.pack(pady=0)

     

        # 캐릭터 이미지 업데이트 - 배경색 변경
        image_path = os.path.join(os.path.dirname(__file__), "templete", "al_model.png")
        self.character_image = Image.open(image_path)
        self.character_image = self.character_image.resize((400, 600))  # Reduced image size
        self.character_photo = ImageTk.PhotoImage(self.character_image)
        self.character_label = tk.Label(
            self.root,
            image=self.character_photo,
            bg='#ADD8E6'
        )
        self.character_label.pack(pady=(50, 10))
        
        # Bottom frame - 배경색 변경
        bottom_frame = tk.Frame(self.root, bg='#ADD8E6')
        bottom_frame.pack(side='bottom', fill='x', pady=(0, 50))
        
        # STT indicator - 배경색 변경
        self.stt_indicator = tk.Label(
            bottom_frame,
            text="음성인식 중...",
            font=("Helvetica", 30),  # Reduced font size from 40
            bg='#ADD8E6',
            fg='red',
            pady=30  # Reduced padding
        )
        self.stt_indicator.pack(pady=10)
        self.stt_indicator.pack_forget()

        # Subtitle label - 배경색 변경
        self.subtitle_label = tk.Label(
            bottom_frame,
            text="",
            font=("Helvetica", 30),  # Reduced font size from 40
            bg='#ADD8E6',
            wraplength=800,  # Reduced wraplength
            pady=30  # Reduced padding
        )
        self.subtitle_label.pack(pady=10)

        # 루트 윈도우 배경색도 변경
        self.root.configure(bg='#ADD8E6')

        self.current_mode = 2

    def show_mode3(self):
        # 기존 위젯들 제거
        for widget in self.root.winfo_children():
            widget.destroy()

        # 이미지 표시
        image_path = os.path.join(os.path.dirname(__file__), "templete", "advertise.png")
        self.character_image = Image.open(image_path)
        
        # 전체 화면 크기 가져오기
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 이미지 크기를 화면 크기에 맞게 조정 (ANTIALIAS 사용)
        self.character_image = self.character_image.resize(
            (screen_width, screen_height), 
            Image.ANTIALIAS
        )
        
        self.character_photo = ImageTk.PhotoImage(self.character_image)
        self.character_label = tk.Label(
            self.root,
            image=self.character_photo,
        )
        self.character_label.pack(fill='both', expand=True)

        self.current_mode = 3
    

    def show_mode4(self):
        # 새 이미지를 먼저 준비
        image_path = os.path.join(os.path.dirname(__file__), "templete", "finish.png")
        self.character_image = Image.open(image_path)
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        self.character_image = self.character_image.resize(
            (screen_width, screen_height), 
            Image.ANTIALIAS
        )
        
        self.character_photo = ImageTk.PhotoImage(self.character_image)
        
        # 새 라벨 생성
        new_label = tk.Label(
            self.root,
            image=self.character_photo,
            bg='#dce8d5'  # 배경색을 기존 배경색과 동일하게 설정
        )
        
        # 기존 위젯들 제거하기 전에 새 라벨을 준비
        new_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # 기존 위젯들 제거
        for widget in self.root.winfo_children():
            if widget != new_label:  # 새로 만든 라벨은 남기고 나머지만 제거
                widget.destroy()
                
        self.character_label = new_label
        self.current_mode = 4



    def switch_mode(self, mode):
        if mode == 2 and self.current_mode != 2:
            self.show_mode2()
        elif mode == 1 and self.current_mode != 1:
            self.show_mode1()
        elif mode == 3 and self.current_mode != 3:
            self.show_mode3()
        elif mode == 4 and self.current_mode != 4:
            self.show_mode4()

    def run(self):
        if self.root is None:
            self.create_window()
        self.root.mainloop()

    def _on_closing(self):
        # Reset screen rotation before closing
        try:
            subprocess.run(['xrandr', '-o', 'normal'])
        except Exception as e:
            print(f"Failed to reset screen rotation: {e}")
            
        if self.root:
            self.root.destroy()
            self.root = None

    def show_stt_indicator(self):
        if self.stt_indicator:
            self.stt_indicator.pack()

    def hide_stt_indicator(self):
        if self.stt_indicator:
            self.stt_indicator.pack_forget()

    def update_subtitle(self, text):
        if self.subtitle_label:
            self.subtitle_label.config(text=text)
    
    def clear_subtitle(self):
        if self.subtitle_label:
            self.subtitle_label.config(text="")

class AudioSTTNode(Node):
    def __init__(self):
        super().__init__('audio_stt_node')
       
        self.running = True # 모든 multi thread를 정지시키기 위한 플래그
        self.model_name = 'tiny'
        self.is_mode3 = False  # Add this flag
        self.follow_flag = False

        # Initialize from imported classes
        self.audio_record = Audio_record()
        self.model = Custom_faster_whisper()
        self.model.set_model(self.model_name)  
        self.tts = Custom_TTS()
        self.tts.set_model(language='ko')
        
        
        # 시스템 메시지 설정
       
        self.target_locations = ['응급실', '수납', '접수', '편의점', '화장실']
        self.system_message = (
            "당신은 병원 안내 로봇입니다. 사람들은 정보성 질문이나 길 안내 질문을 할 수 있습니다. "
            "정보성 질문에 대해서는 간단하고 명확한 답변을 제공하세요. "
            "목적지를 찾는 질문의 경우, 지정된 위치에의 안내만 가능합니다. "
            "다른 장소에 대한 길 안내 요청에는 '현재 해당 위치로의 안내 서비스는 준비중입니다'라고 답변하세요. "
            "응급실을 물어보면 '응급실은 좌회전 후 직진하면 위치해 있습니다.'라고 답변하세요. "
            "수ㄴ 및 접수 장소를 물어보면 '수납 및 접수 장소는 좌회전 후 직진하면 위치해 있습니다.'라고 답변하세요. "
            "편의점을 물어보면 '편의점은 직진후 좌회전으로 가면 위치해있습니다.'라고 답변하세요. "
            "화장실을 물어보면 '화장실은 직진후 좌회전으로 가면 위치해있습니다.'라고 답변하세요."
        )
        self.running = True # 모든 multi thread를 정지시키기 위한 플래그
        self.model_name = 'tiny'
        self.location_mapping = {
            '응급실': 0,  
            '수납': 1,
            '접수': 1,
            '편의점': 2,
            '화장실': 3
        }
        
        # 타이머 수정 - 1초마다 체크
        self.create_timer(1.0, self.timer_callback)
        self.is_stt_running = False

        self.gui = SttGui()
        self.gui_thread = None

        # 구독자 추가
        self.subscription = self.create_subscription(
            Int32,
            'search_complete',
            self.stt_callback,
            10)

        self.advertise_sub = self.create_subscription(
            Int32,
            'pantilt_search_complete',
            self.advertise_callback,
            10)
        
        self.advertise_sub = self.create_subscription(
            Int32,
            'nav_complet',
            self.nav_callback,
            10)

        self.audio_recorder = self.audio_record  # Alias for consistency

        # Change publisher type from String to Int32
        self.location_pub = self.create_publisher(Int32, 'location_guidance', 10)

        self.nav_start_pub = self.create_publisher(Int32, 'nav_start', 10)

        self.guide_finish_pub = self.create_publisher(Int32, 'finish_detection', 10)
    
    # 타이머 콜백 함수
    def timer_callback(self):  # self 매개변수 추가
        # is_stt_running이 False일 때만 새로운 STT 프로세스 시작
        if not self.is_stt_running:
            self.is_stt_running = True
            self.gui_thread = threading.Thread(target=self.gui.run)
            self.gui_thread.start()

    def stt_callback(self, msg):
       
        self.gui.root.after(0, lambda: self.start_mode2())
    
    
    def advertise_callback(self, msg):

        self.follow_flag = True
    
    def nav_callback(self, msg):
        def switch_and_speak():
            self.gui.switch_mode(4)
            end_message = "안내를 종료합니다"
            
            def after_speech():
                # 3초 대기 후 guide_finish 토픽 발행 및 모드1로 전환
                time.sleep(3)
                guide_finish_msg = Int32()
                guide_finish_msg.data = 1
                self.guide_finish_pub.publish(guide_finish_msg)
                self.gui.root.after(0, lambda: self.gui.switch_mode(1))

            # 자막 업데이트 후 음성 재생, 그 후 after_speech 실행
            self.gui.root.after(50, lambda: [
                self.gui.update_subtitle(end_message),
                self.speak(end_message),
                threading.Thread(target=after_speech, daemon=True).start()
            ])
            
        self.gui.root.after(0, switch_and_speak)
            
                                
          

    def start_mode2(self):
        self.is_mode3 = False  # Reset the flag when entering mode 2
        self.gui.switch_mode(2)
        # Start the STT process after showing mode 2
        self.gui.root.after(500, self.start_stt_process)

    def start_stt_process(self):
        def run_stt():
            # 먼저 자막을 업데이트한 후 음성 재생
            greeting = "안녕하세요 한양로봇 안내도우미 알쏭이 입니다. 궁금하신 사항이 있다면 화면 하단에 음성인식 표시가 생긴 후에 질문해주세요"
            self.gui.root.after(0, lambda: self.gui.update_subtitle(greeting))
            self.speak(greeting)

            while self.running:
                # Stop STT if in mode 3
                if self.is_mode3:
                    break

                # Static variable to track iterations
                if not hasattr(self, '_first_iteration'):
                    self._first_iteration = True
                elif self._first_iteration:
                    speak_again = "길 안내가 필요하시면 화면 하단에 음성인식 표시가 생긴 후에 질문해주세요"
                    self.gui.root.after(0, lambda: self.gui.update_subtitle(speak_again))
                    self.speak(speak_again)
                    self._first_iteration = True

                self.gui.root.after(0, self.gui.show_stt_indicator)
                self.audio_recorder.vad_sec = 3
                self.audio_recorder.record_start()
                
                # 녹음 시작 후 3초 타이머 시작
                start_time = time.time()
                while time.time() - start_time < 3 and self.audio_recorder.recording:
                    if (self.audio_recorder.buffer and 
                        len(self.audio_recorder.buffer[-1]) == self.audio_recorder.chunk_size):
                        chunk = self.audio_recorder.buffer[-1]
                        if self.audio_recorder._vad(chunk, self.audio_recorder.sample_rate):
                            self.audio_recorder.vad_sec = 1
                            break
                    time.sleep(0.1)
                
                if not self.audio_recorder.buffer:
                    self.audio_recorder.recording = False
                    self.gui.root.after(0, self.gui.hide_stt_indicator)
                    continue

                audio_result = self.audio_recorder.record_stop(0.5)
                self.gui.root.after(0, self.gui.hide_stt_indicator)

                if not audio_result['audio_denoise'].size:
                    continue

                _, result_text, _ = self.model.run(audio_result['audio_denoise'], language='ko')
                
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {'role': 'system', 'content': self.system_message},
                        {'role': 'user', 'content': result_text}
                    ]
                )
                chatgpt_reply = response['choices'][0]['message']['content'].strip()
                self.gui.root.after(0, lambda: self.gui.update_subtitle(chatgpt_reply))
                self.speak(chatgpt_reply)

                # 위치 안내 처리
                for location in self.target_locations:
                    if location in chatgpt_reply:
                        guidance_message = f'{location}로 안내를 시작합니다.'
                        self.gui.root.after(0, lambda: self.gui.update_subtitle(guidance_message))
                        self.speak(guidance_message)
                        self.publish_location(location)
                        self._first_iteration = True
                        # Wait for advertise_start topic

                        while(1):

                            if self.follow_flag:
                                
                                speak_follow = "저를 따라오세요!"
                                self.gui.root.after(0, lambda: self.gui.update_subtitle(speak_follow))
                                self.speak(speak_follow)

                                nav_start_msg = Int32()
                                nav_start_msg.data = 1
                                self.nav_start_pub.publish(nav_start_msg)

                                # Set mode3 flag before switching mode
                                self.is_mode3 = True
                                self.gui.root.after(0, lambda: self.gui.switch_mode(3))
                                self.follow_flag = False

                                return
        
                time.sleep(2)  # 다음 음성인식 시작 전 대기

        threading.Thread(target=run_stt, daemon=True).start()

    # 위치 정보 발행 함수
    def publish_location(self, location):
        msg = Int32()
        msg.data = self.location_mapping.get(location, 0)  # Default to 0 if location not found
        self.location_pub.publish(msg)

    #stt 구동 보조 함수

    def speak(self, text):
        output_path = self.tts.make_speech(text)
        if output_path:
            try:
                # MP3를 WAV로 변환 (simpleaudio는 WAV만 지원)
                wav_path = output_path.replace('.mp3', '.wav')
                os.system(f'ffmpeg -y -i {output_path} -af "adelay=1000|1000" {wav_path}')

                
                # 약간의 지연시간 추가
                time.sleep(0.1)  # 100ms 대기
                
                # WAV 파일 재생
                wave_obj = sa.WaveObject.from_wave_file(wav_path)
                # 재생 시작 전 약간의 준비 시간
                time.sleep(0.05)  # 50ms 대기
                play_obj = wave_obj.play()
                play_obj.wait_done()  # 재생이 끝날 때까지 대기
                
                # 임시 WAV 파일 삭제
                os.remove(wav_path)
                os.remove(output_path)  # MP3 파일도 삭제
                
                # TTS 재생이 끝난 후 자막 제거
                if self.gui.root:
                    self.gui.root.after(0, self.gui.clear_subtitle)
            except Exception as e:
                print(f"Audio playback error: {e}")

    def _on_closing(self):
        self.running = False
        print('프로그램이 종료됩니다')
        self.root.destroy()

def main(args=None):
    rclpy.init(args=args)
    node = AudioSTTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


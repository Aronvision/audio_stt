import os
import threading
import time
import wave
import numpy as np
import speech_recognition as sr
import webrtcvad
from faster_whisper import WhisperModel
import openai
import shutil
from gtts import gTTS
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import ast  # 문자열로 된 딕셔너리를 파싱하기 위해 추가

class Audio_record:
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_duration_ms = 30
        self.vad_sec = 1
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        self.buffer = []
        self.recording = False

        self.vad = webrtcvad.Vad(1)
        self.adjust_noise()
        print('Audio_record 초기화 성공')

    def adjust_noise(self):
        print('주변 소음에 맞게 조정 중...')
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.recognizer.energy_threshold += 100

    def record_start(self):
        if not self.recording:
            self.record_thread = threading.Thread(target=self._record_start)
            self.record_thread.start()

    def _record_start(self):
        self.recording = True
        self.buffer = []
        no_voice_target_cnt = (self.vad_sec*1000)
        no_voice_cnt = 0
        with self.microphone as source:
            while self.recording:
                chunk = source.stream.read(self.chunk_size)
                self.buffer.append(chunk)
                if self._vad(chunk, self.sample_rate):
                    no_voice_cnt = 0
                else:
                    no_voice_cnt += self.chunk_duration_ms
                if no_voice_cnt >= no_voice_target_cnt:
                    self.recording = False

    def _vad(self, chunk, sample_rate):
        if isinstance(chunk, bytes):
            chunk = np.frombuffer(chunk, dtype=np.int16)
        if len(chunk) != self.chunk_size:
            raise ValueError("Chunk size must be exactly 10ms, 20ms, or 30ms")
        return self.vad.is_speech(chunk.tobytes(), sample_rate)
        
    def record_stop(self):
        self.recording = False
        self.record_thread.join()
        audio_data = self._buffer_to_numpy(self.buffer, self.microphone.SAMPLE_RATE)
        return {'audio': audio_data, 'sample_rate': self.microphone.SAMPLE_RATE}

    def _buffer_to_numpy(self, buffer, sample_rate):
        audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        return audio_data

class Custom_faster_whisper:
    def __init__(self):
        try: os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
        except: pass
        try: os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        except: pass
        print('Cumtom_faster_whisper 초기화 성공')

    def set_model(self, model_name):
        model_list = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3', 'large']
        if model_name not in model_list:
            model_name = 'tiny'
            print('모델 이름 잘못됨. base로 설정. ���래 모델 중 한가지 선택')
            print(model_list)
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        return model_name

    def run(self, audio, language=None):
        start = time.time()
        segments, info = self.model.transcribe(audio, beam_size=5, word_timestamps=True, language=language)
        dic_list = []
        for segment in segments:
            if segment.no_speech_prob > 0.6:
                continue
            for word in segment.words:
                dic_list.append([word.word, round(word.start, 2), round(word.end, 2)])
        self.spent_time = round(time.time()-start, 2)
        result_txt = ''.join([w[0] for w in dic_list])
        print(result_txt)
        return dic_list, result_txt, self.spent_time

class Custom_TTS:
    def __init__(self):
        self.result_cnt = 0
        self.output_path = "./tts_output"
        # 기존 출력 폴더 삭제 후 새로 생성
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

    def set_model(self, language='ko'):
        '''
        TTS 언어 설정
        language: 언어 코드 (ko: 한국어, en: 영어, ja: 일본어 등)
        '''
        self.language = language
        print('TTS 설정 완료')

    def make_speech(self, text):
        try:
            output_file = f"{self.output_path}/result_{self.result_cnt}.mp3"
            tts = gTTS(text=text, lang=self.language)
            tts.save(output_file)
            print('TTS 변환 완료')
            self.result_cnt += 1
            return output_file
        except Exception as e:
            print(f"TTS 변환 실패: {e}")
            return None

class AudioSTTNode(Node):
    def __init__(self):
        super().__init__('audio_stt_node')
        self.location_pub = self.create_publisher(String, 'target_location', 10)
        
        # 새로운 subscriber 추가
        self.detection_sub = self.create_subscription(
            String,
            'human_detection_results',
            self.detection_callback,
            10
        )
        self.image_sub = self.create_subscription(
            Image,
            'annotated_frame',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        
        # 기존 객체들 초기화
        self.audio_recorder = Audio_record()
        self.whisper = Custom_faster_whisper()
        self.whisper.set_model('tiny')  
        self.tts = Custom_TTS()
        self.tts.set_model(language='ko')
        
        # 시스템 메시지 설정
        self.system_message = (
            "당신은 병원 안내 로봇입니다. 사람들은 정보성 질문이나 길 안내 질문을 할 수 있습니다. "
            "정보성 질문에 대해서는 적절한 답변을 제공하세요. "
            "길 안내 질문의 경우, 응급실, 수납 및 접수 장소, 편의점, 화장실로의 안내만 가능합니다. "
            "다른 장소에 대한 길 안내 요청에는 '서비스를 준비중입니다'라고 답변하세요."
        )
        self.target_locations = ['응급��', '수납', '접수', '편의점', '화장실']
        self.location_mapping = {
            '응급실': '0',
            '수납': '1',
            '접수': '1',
            '편의점': '2',
            '화장실': '3'
        }
        
        # 바운딩 박스 추적을 위한 변수들
        self.last_center = None
        self.stable_start_time = None
        self.is_person_stable = False
        self.movement_threshold = 50  # 픽셀 단위의 움직임 임계값
        self.stable_threshold = 3.0   # 안정화 시간 임계값 (초)
        
        # STT 상태 관리
        self.is_stt_running = False

    def calculate_center(self, bbox):
        # [xmin, ymin, xmax, ymax] 형식의 바운딩 박스에서 중앙값 계산
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        return (x_center, y_center)

    def is_stable_position(self, current_center):
        if self.last_center is None:
            self.last_center = current_center
            self.stable_start_time = time.time()
            return False

        # 이전 중앙값과 현재 중앙값의 거리 계산
        distance = np.sqrt((current_center[0] - self.last_center[0])**2 + 
                         (current_center[1] - self.last_center[1])**2)

        if distance > self.movement_threshold:
            # 움직임이 임계값을 초과하면 타이머 재설정
            self.stable_start_time = time.time()
            self.last_center = current_center
            return False

        # 안정화 시간 체크
        stable_duration = time.time() - self.stable_start_time
        self.last_center = current_center
        return stable_duration >= self.stable_threshold

    def publish_location(self, location):
        msg = String()
        msg.data = self.location_mapping.get(location, '0')
        self.location_pub.publish(msg)
        self.get_logger().info(f'Published location: {msg.data}')

    def detection_callback(self, msg):
        try:
            # 문자열을 딕셔너리로 변환
            self.get_logger().info("Received message on 'human_detection_results' topic")
            detection_data = ast.literal_eval(msg.data)
            
            if detection_data["detected"] and not self.is_stt_running and detection_data["bounding_boxes"]:
                # 첫 번째 감지된 사람의 바운딩 박스 사용
                bbox = detection_data["bounding_boxes"][0]  # 이미 리스트 형태
                current_center = self.calculate_center(bbox)
                
                if self.is_stable_position(current_center):
                    if not self.is_person_stable:
                        self.is_person_stable = True
                        # STT 프로세스 시작
                        self.start_stt_process()
                else:
                    self.is_person_stable = False
        except Exception as e:
            self.get_logger().error(f'Detection callback error: {str(e)}')

    def start_stt_process(self):
        self.is_stt_running = True
        self.get_logger().info("사람이 안정적으로 감지되어 음성 인식을 시작합니다...")
        
        # 음성 인식 프로세스
        print("말씀하세요... (Enter 입력하면 음성인식 종료)")
        self.audio_recorder.record_start()
        try:
            input("녹음을 멈추려면 Enter를 누르세요...")
        except KeyboardInterrupt:
            pass
        
        audio_result = self.audio_recorder.record_stop()
        print("음성을 텍스트로 변환 중...")
        _, result_text, spent_time = self.whisper.run(audio_result['audio'], language='ko')
        
        # ChatGPT 요청 및 응답 처리
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': result_text}
                ]
            )
            chatgpt_reply = response['choices'][0]['message']['content'].strip()
            
            # TTS 변환 및 재생
            output_path = self.tts.make_speech(chatgpt_reply)
            if output_path:
                os.system(f'play {output_path}')
            
            # 위치 정보 처리
            for location in self.target_locations:
                if location in result_text:
                    guidance_message = f'{location}로 안내를 시작합니다.'
                    output_path = self.tts.make_speech(guidance_message)
                    if output_path:
                        os.system(f'play {output_path}')
                    self.publish_location(location)
                    break
                    
        except Exception as e:
            self.get_logger().error(f'처리 중 오류 발생: {str(e)}')
        
        self.is_stt_running = False

    def image_callback(self, msg):
        self.get_logger().info("Received message on 'annotated_frame' topic")
        try:
            # ROS Image 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 이미지 표시 (선택사항)
            cv2.imshow("Human Detection", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'이미지 처리 중 에러 발생: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = AudioSTTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



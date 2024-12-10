# ws_server.py
import asyncio
import websockets
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # 필요에 따라 메시지 타입 변경 가능

class WebSocketServer(Node):
    def __init__(self):
        super().__init__('websocket_server')
        self.publisher_ = self.create_publisher(String, 'topic_humble', 10)
        self.get_logger().info('WebSocket 서버 초기화 완료')

    async def handler(self, websocket, path):
        self.get_logger().info(f"클라이언트 연결됨: {websocket.remote_address}")
        try:
            async for message in websocket:
                self.get_logger().info(f"수신된 메시지: {message}")
                data = json.loads(message)
                ros_msg = String()
                ros_msg.data = data['data']['message']
                self.publisher_.publish(ros_msg)
                self.get_logger().info(f"ROS 2 토픽에 메시지 퍼블리시: {ros_msg.data}")
        except websockets.exceptions.ConnectionClosed:
            self.get_logger().info(f"클라이언트 연결 종료: {websocket.remote_address}")

    def run_server(self):
        start_server = websockets.serve(self.handler, "0.0.0.0", 8765)
        self.get_logger().info("WebSocket 서버 시작: ws://0.0.0.0:8765")
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

def main(args=None):
    rclpy.init(args=args)
    ws_server = WebSocketServer()
    try:
        ws_server.run_server()
    except KeyboardInterrupt:
        ws_server.get_logger().info("서버 종료")
    finally:
        ws_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

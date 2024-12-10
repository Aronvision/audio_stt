import cv2
import numpy as np

def main():
    # 검은색 이미지 생성
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 흰색 사각형 그리기
    cv2.rectangle(img, (100, 100), (540, 380), (255, 255, 255), 3)
    
    # 이미지 표시
    cv2.imshow("Test Window", img)
    
    # 키 입력 대기
    cv2.waitKey(0)
    
    # 창 닫기
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

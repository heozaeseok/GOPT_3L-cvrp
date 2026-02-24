import socket
import json
import time

def run_client():
    print("[*] 테스트 클라이언트를 시작합니다. 'q' 입력 시 종료.")
    
    while True:
        user_input = input("\n[Input] 경로를 입력하세요 (예: 5 7 4): ")
        if user_input.lower() == 'q': break
        
        # try-except 없이 조건문으로 숫자 여부 검증
        is_valid = True
        for x in user_input.split():
            if not x.isdigit():
                is_valid = False
                break
                
        if not is_valid:
            print("[-] 양의 정수만 입력해주세요.")
            continue

        route = [int(x) for x in user_input.split()]
        start_time = time.time()
        
        # 서버 연결 (연결 실패 시 에러가 터지도록 예외처리 제거)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 9999))
        
        message = json.dumps({"route": route})
        s.sendall(message.encode('utf-8'))
        
        response = s.recv(1024).decode('utf-8')
        s.close()
        
        duration = time.time() - start_time
        print(f"[Result] 서버 응답: {response}")
        print(f"[Time] 소요 시간: {duration:.4f} 초")

if __name__ == "__main__":
    run_client()
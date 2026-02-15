import socket
import json
import time

def run_client():
    print("[*] 테스트 클라이언트를 시작합니다. 'q' 입력 시 종료.")
    
    while True:
        # 1. 터미널에서 경로(route)만 입력받음
        user_input = input("\n[Input] 경로를 입력하세요 (예: 5 7 4): ")
        if user_input.lower() == 'q': break
        
        try:
            route = [int(x) for x in user_input.split()]
        except ValueError:
            print("[-] 숫자만 입력해주세요.")
            continue

        start_time = time.time()
        
        # 2. 서버 연결 및 전송
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(('127.0.0.1', 9999))
                # 파일 ID는 서버가 이미 알고 있으므로 route만 전송
                message = json.dumps({"route": route})
                s.sendall(message.encode('utf-8'))
                
                response = s.recv(1024).decode('utf-8')
                duration = time.time() - start_time
                
                print(f"[Result] 서버 응답: {response}")
                print(f"[Time] 소요 시간: {duration:.4f} 초")
            except ConnectionRefusedError:
                print("[-] 서버 연결 실패. ts_TF_server.py를 실행 중인지 확인하세요.")

if __name__ == "__main__":
    run_client()
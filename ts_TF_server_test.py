import socket
import json
import time

def run_client():
    print("[*] Python 테스트 클라이언트를 시작합니다. 종료하려면 'q' 입력.")
    while True:
        user_input = input("\n[Input] 파일ID와 경로 입력 (예: 1 5 7 4): ")
        if user_input.lower() == 'q': break
        
        parts = user_input.split()
        if not parts: continue
        
        file_id = int(parts[0])
        route = [int(x) for x in parts[1:]]

        start_time = time.time()
        
        # 서버 연결
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(('127.0.0.1', 9999))
                message = json.dumps({"file": file_id, "route": route})
                s.sendall(message.encode('utf-8'))
                
                response = s.recv(1024).decode('utf-8')
                duration = time.time() - start_time
                
                print(f"[Result] 서버 응답: {response}")
                print(f"[Time] 소요 시간: {duration:.4f} 초")
            except ConnectionRefusedError:
                print("[-] 서버가 꺼져 있습니다. ts_TF_server.py를 먼저 실행하세요.")

if __name__ == "__main__":
    run_client()
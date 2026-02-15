#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
    #include <winsock2.h>
    #pragma comment(lib, "ws2_32.lib")
    #define close closesocket
#else
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <unistd.h>
#endif

void check_route_interactive() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    char input[512];
    
    while (1) {
        printf("\n[Input] 경로를 입력하세요 (예: 1 5 7 4) / 종료하려면 'q' 입력: ");
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0; // 줄바꿈 제거

        if (strcmp(input, "q") == 0) break;
        if (strlen(input) == 0) continue;

        // --- 시간 측정 시작 ---
        clock_t start_time = clock();

        // 소켓 생성 및 연결
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in server;
        server.sin_family = AF_INET;
        server.sin_port = htons(9999);
        server.sin_addr.s_addr = inet_addr("127.0.0.1");

        if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
            printf("[-] Connection Failed. 서버가 켜져 있는지 확인하세요.\n");
            continue;
        }

        // 입력된 숫자를 JSON 배열 형식으로 변환
        char route_json[1024] = "";
        char *ptr = strtok(input, " ");
        while (ptr != NULL) {
            strcat(route_json, ptr);
            ptr = strtok(NULL, " ");
            if (ptr != NULL) strcat(route_json, ", ");
        }

        char message[2048];
        // file_id는 테스트용으로 1로 고정
        sprintf(message, "{\"file\": 1, \"route\": [%s]}", route_json);

        // 데이터 전송
        send(sock, message, strlen(message), 0);

        // 결과 수신
        char buffer[2048] = {0};
        recv(sock, buffer, sizeof(buffer), 0);

        // --- 시간 측정 종료 ---
        clock_t end_time = clock();
        double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("[Result] %s\n", buffer);
        printf("[Time] 소요 시간: %.4f 초\n", duration);

        close(sock);
    }

#ifdef _WIN32
    WSACleanup();
#endif
}

int main() {
    printf("[*] 경로 판단 C 클라이언트 테스트를 시작합니다.\n");
    check_route_interactive();
    return 0;
}
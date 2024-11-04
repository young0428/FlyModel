#!/bin/bash

QUEUE_FILE="/myhome/FlyModel/task_queue.txt"
LOCK_FILE="/myhome/FlyModel/task_queue.lock"

# 작업을 큐에 추가
echo "$1" >> "$QUEUE_FILE"

# 큐가 비어 있지 않으면 작업 실행
(
    flock -n 200 || exit 1

    while IFS= read -r task; do
        echo "Executing: $task"
        if eval "$task"; then
            # 작업이 성공적으로 완료된 경우에만 큐에서 제거
            sed -i '1d' "$QUEUE_FILE"  # 첫 번째 줄 삭제
        else
            echo "Task failed: $task"
            break  # 실패한 경우 루프를 종료하여 나머지 작업은 대기 상태로 유지
        fi
    done < "$QUEUE_FILE"
) 200>"$LOCK_FILE"

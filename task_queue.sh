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
        eval "$task"
        sed -i '1d' "$QUEUE_FILE"  # 첫 번째 줄 삭제
    done < "$QUEUE_FILE"
) 200>"$LOCK_FILE" 
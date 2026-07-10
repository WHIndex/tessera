#!/bin/bash
# Monitor GPU 0 and unload llama3.3 if it gets loaded (to prevent OOM for BGE + Router)
LOG=/tmp/gpu0_monitor.log
echo "$(date): GPU0 monitor started, PID=$$" >> "$LOG"
while true; do
  # Check if llama3.3 is loaded on old Ollama
  resp=$(curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null)
  if echo "$resp" | grep -q '"name": "llama3.3:latest"'; then
    echo "$(date): llama3.3 detected, unloading..." >> "$LOG"
    curl -s --max-time 10 -X POST http://localhost:11434/api/generate \
      -d '{"model": "llama3.3:latest", "prompt": "", "stream": false, "keep_alive": 0}' > /dev/null 2>&1
    echo "$(date): unload request sent" >> "$LOG"
  fi
  sleep 30
done

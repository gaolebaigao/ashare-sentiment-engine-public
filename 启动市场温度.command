#!/bin/zsh

set -e
unsetopt BG_NICE

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "正在启动市场温度系统，请稍候……"
echo "启动后请保持这个窗口开启。"
echo

# 旧版服务或上次未关闭的窗口可能仍占用默认端口。自动选择空闲端口，
# 避免把旧页面误认为本次刚启动的新系统。
PORT=8765
while /usr/bin/nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1; do
  PORT=$((PORT + 1))
  if (( PORT > 8795 )); then
    echo "找不到可用端口（8765-8795），请关闭旧的市场温度终端窗口后重试。"
    exit 1
  fi
done

URL="http://127.0.0.1:${PORT}/"
echo "本次访问地址：$URL"

python -m ashare_sentiment web --no-open --port "$PORT" &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

for attempt in {1..60}; do
  if /usr/bin/curl -fsS "$URL" >/dev/null 2>&1; then
    echo "系统已启动，正在打开浏览器……"
    /usr/bin/open "$URL"
    wait "$SERVER_PID"
    exit $?
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID"
    exit $?
  fi
  sleep 0.5
done

echo "启动超时，请关闭窗口后重新双击启动文件。"
exit 1

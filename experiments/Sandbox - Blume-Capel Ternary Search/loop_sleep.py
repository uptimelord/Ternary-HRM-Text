import json
import sys
import time

if len(sys.argv) < 2:
    raise SystemExit("usage: loop_sleep.py <seconds> <prompt>")
seconds = int(sys.argv[1])
prompt = sys.argv[2]
time.sleep(seconds)
print(f"AGENT_LOOP_WAKE_blume_capel {json.dumps({'prompt': prompt})}", flush=True)

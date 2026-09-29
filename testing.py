import psutil, requests, time

def get_worker():
    procs = [p for p in psutil.process_iter(['pid', 'memory_info', 'cmdline'])
             if p.info['cmdline'] and 'uvicorn' in ' '.join(p.info['cmdline']).lower()]
    return max(procs, key=lambda p: p.info['memory_info'].rss)

def report(label):
    p = get_worker()
    print(f"{label}: PID {p.info['pid']}  RSS {p.info['memory_info'].rss / 1024 / 1024:.0f} MB")
    return p.info['pid']

pid_1 = report("Before anything")

requests.get("http://localhost:8000/ping")
time.sleep(1)
pid_2 = report("After /ping")

login = requests.post("http://localhost:8000/api/auth/login",
                      data={"username": "collegekafunda2018@gmail.com", "password": "Rishu@0200"})
token = login.json()["access_token"]
requests.post("http://localhost:8000/api/query", json={"question": "forecast for TBP-001"},
              headers={"Authorization": f"Bearer {token}"})
time.sleep(1)
pid_3 = report("After forecast query")

requests.post("http://localhost:8000/api/query", json={"question": "who is the best supplier for RSH-001?"},
              headers={"Authorization": f"Bearer {token}"})
time.sleep(1)
pid_4 = report("After supplier query")

if len({pid_1, pid_2, pid_3, pid_4}) > 1:
    print("\n⚠️  PID changed during the run — the server restarted mid-test (likely --reload).")
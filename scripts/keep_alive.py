import time
import argparse
import urllib.request
from datetime import datetime


def ping(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            status = r.status
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] ✓ {url} → {status}")
            return True
    except Exception as e:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] ✗ {url} → {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/ping",
                        help="URL to ping (default: localhost)")
    parser.add_argument("--interval", type=int, default=840,
                        help="Interval in seconds (default: 840 = 14 min)")
    args = parser.parse_args()

    print(f"🏓 Keep-alive pinging {args.url} every {args.interval//60} min")
    
    while True:
        ping(args.url)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
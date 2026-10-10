"""Generate examples/nginx_sample.log: normal traffic plus one brute-force burst."""

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

rng = random.Random(7)
START = datetime(2026, 10, 7, 10, 0, 0, tzinfo=UTC)
PATHS = ["/", "/index.html", "/products", "/products/42", "/about", "/api/items", "/static/app.js"]
AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) Safari/605.1",
]

events: list[tuple[int, str]] = []


def add(offset: int, ip: str, method: str, path: str, status: int, size: int, agent: str) -> None:
    stamp = (START + timedelta(seconds=offset)).strftime("%d/%b/%Y:%H:%M:%S +0000")
    line = f'{ip} - - [{stamp}] "{method} {path} HTTP/1.1" {status} {size} "-" "{agent}"'
    events.append((offset, line))


# Normal traffic: 25 clients, 12 minutes, 1-3 requests per client per minute.
for n in range(1, 26):
    ip = f"198.51.100.{n}"
    agent = rng.choice(AGENTS)
    for minute in range(12):
        for _ in range(rng.randint(1, 3)):
            status = rng.choices([200, 304, 404], weights=[90, 8, 2])[0]
            add(minute * 60 + rng.randint(0, 59), ip, "GET", rng.choice(PATHS),
                status, rng.randint(300, 9000), agent)

# Brute force: 90 failed logins from one address inside ~90 seconds.
for i in range(90):
    add(300 + i, "203.0.113.50", "POST", f"/login?user=admin&password=Passw0rd{i}",
        401, 187, "python-requests/2.31")

events.sort(key=lambda item: item[0])
target = Path(__file__).with_name("nginx_sample.log")
target.write_text("\n".join(line for _, line in events) + "\n", encoding="utf-8")
print(f"wrote {len(events)} lines to {target}")
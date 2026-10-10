"""Synthetic nginx log builders for command-level tests."""

from __future__ import annotations

import random
from datetime import timedelta
from pathlib import Path

from .features.helpers import T0

PATHS = ["/", "/about", "/api/items", "/static/app.js"]


def nginx_line(offset_s: float, ip: str, path: str, status: int, nbytes: int, ua: str) -> str:
    stamp = (T0 + timedelta(seconds=offset_s)).strftime("%d/%b/%Y:%H:%M:%S +0000")
    return f'{ip} - - [{stamp}] "GET {path} HTTP/1.1" {status} {nbytes} "-" "{ua}"'


def normal_lines(n_ips: int = 30, minutes: int = 10, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    lines = []
    for ip in range(n_ips):
        for minute in range(minutes):
            status = 404 if rng.random() < 0.05 else 200
            lines.append(
                nginx_line(
                    minute * 60 + ip,
                    f"10.0.0.{ip}",
                    rng.choice(PATHS),
                    status,
                    rng.randint(200, 2000),
                    f"browser-{ip % 3}",
                )
            )
    return lines


def attack_lines(ip: str = "9.9.9.9", start: float = 300.0, n: int = 80) -> list[str]:
    return [
        nginx_line(start + i * 0.5, ip, "/login?password=hunter2", 401, 512, "curl")
        for i in range(n)
    ]


def write_log(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

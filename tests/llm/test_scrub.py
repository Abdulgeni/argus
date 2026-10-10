import pytest

from argus.llm.scrub import REDACTED, scrub

CASES = [
    ("Authorization: Bearer abc123.def456", "abc123"),
    ("authorization=Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ("GET /x?password=hunter2&a=1 HTTP/1.1", "hunter2"),
    ('api_key: "sk-ABCDEFGHIJKLMNOPQRSTUV"', "ABCDEFGHIJKLMNOPQRSTUV"),
    ("api-key=zz99yy88", "zz99yy88"),
    ('{"token":"xyz987","user":"bob"}', "xyz987"),
    ("access_token=abc; other=1", "abc;"),
    ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4f here", "eyJhbGci"),
    ("key AKIAIOSFODNN7EXAMPLE leaked", "AKIAIOSFODNN7EXAMPLE"),
    ("https://bob:s3cret@db.example.com/path", "s3cret"),
    ("Cookie: session=abcdef; theme=dark", "abcdef"),
    ("secret='top secret value' next", "top secret value"),
    ("token ghp_" + "a" * 36, "a" * 36),
    ("client_secret=Zm9vYmFy&grant_type=x", "Zm9vYmFy"),
]


@pytest.mark.parametrize(("text", "secret"), CASES)
def test_secrets_removed(text: str, secret: str) -> None:
    cleaned = scrub(text)
    assert secret not in cleaned
    assert REDACTED in cleaned


def test_context_around_secret_is_kept() -> None:
    cleaned = scrub("GET /x?password=hunter2&a=1 HTTP/1.1")
    assert cleaned == f"GET /x?password={REDACTED}&a=1 HTTP/1.1"
    assert scrub('{"token":"xyz987","user":"bob"}').endswith('"user":"bob"}')


@pytest.mark.parametrize(
    "text",
    [
        '203.0.113.9 - - [09/Oct/2026:09:14:02 +0000] "GET /index.html HTTP/1.1" 200 512 "-" "ua"',
        "tokenizer=fast passenger=3 secretary=bob",
        "",
    ],
)
def test_innocent_lines_unchanged(text: str) -> None:
    assert scrub(text) == text


def test_idempotent() -> None:
    once = scrub("password=abc Authorization: Bearer xyz")
    assert scrub(once) == once

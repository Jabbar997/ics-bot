"""Read-only local web dashboard for ICS.

A second front end next to the Telegram bot, built on the same
:class:`CommandService`, so the dashboard and the bot can never disagree.

Deliberately narrow:
* **Read-only.** Only the report commands in :data:`READ_COMMANDS` are served.
  ``stop`` / ``resume`` are not routable, and only GET is accepted.
* **Local by default.** :func:`serve` binds to loopback only. A non-loopback
  bind (the hosted worker) is refused unless a password is set, or the server
  is explicitly ``locked`` (health check only, no data).
* **Password-gated when hosted.** HTTP Basic over the platform's HTTPS, with
  constant-time comparison and a per-client lockout after repeated failures.
* **No new dependencies.** Standard library only.
"""
from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app.config import Config
from app.db.database import session_scope
from app.db.repositories import PositionRepository
from app.risk.kill_switch import KillSwitchManager
from app.telegram.commands import SAFETY, CommandService

log = logging.getLogger(__name__)

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
MAX_FAILURES = 5          # wrong passwords allowed ...
FAILURE_WINDOW_S = 300    # ... within this many seconds, then the client is locked out
INDEX_HTML = Path(__file__).with_name("index.html")

# Report commands the dashboard may run. Control commands are excluded on purpose.
READ_COMMANDS: Tuple[str, ...] = (
    "status",
    "portfolio",
    "positions",
    "today",
    "weekly",
    "performance",
    "rejected",
    "audit",
    "learning",
    "health",
    "rules",
    "watchlist",
    "kill",
)


def build_summary(config: Config) -> Dict[str, object]:
    """Structured numbers for the dashboard cards."""
    from app.paper.portfolio import Portfolio

    with session_scope() as s:
        pf = Portfolio(s, config.initial_capital)
        cash = float(pf.calculate_cash())
        invested = float(pf.calculate_invested_value())
        total = float(pf.calculate_total_value())
        ret = float(pf.total_return_pct())
        ks = KillSwitchManager(s)
        frozen = ks.is_frozen()
        level = ks.active_level().value
        positions = []
        for p in PositionRepository(s).open_positions():
            cur = float(p.current_price or p.entry_price)
            positions.append(
                {
                    "ticker": p.ticker,
                    "entry": float(p.entry_price),
                    "current": cur,
                    "pnl_pct": cur / float(p.entry_price) - 1.0,
                    "dqs": p.dqs_at_entry,
                }
            )
    return {
        "mode": config.mode,
        "safety": SAFETY,
        "initial_capital": float(config.initial_capital),
        "cash": cash,
        "invested": invested,
        "total": total,
        "return_pct": ret,
        "frozen": frozen,
        "kill_level": level,
        "positions": positions,
    }


def _client_id(handler: BaseHTTPRequestHandler) -> str:
    """Best-effort client key. Behind a proxy the LAST X-Forwarded-For entry is
    the one the platform's own proxy appended; earlier ones are client-supplied."""
    fwd = handler.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[-1].strip()
    return handler.client_address[0]


def make_handler(config: Config, password: Optional[str] = None, locked: bool = False) -> type:
    service = CommandService(config)
    failures: Dict[str, List[float]] = {}
    lock = threading.Lock()

    def _recent_failures(client: str) -> List[float]:
        now = time.monotonic()
        recent = [t for t in failures.get(client, []) if now - t < FAILURE_WINDOW_S]
        failures[client] = recent
        return recent
    routes: Dict[str, Callable[[], object]] = {
        "summary": lambda: build_summary(config),
    }
    for name in READ_COMMANDS:
        routes[name] = (lambda n=name: {"text": getattr(service, n)()})

    class Handler(BaseHTTPRequestHandler):
        server_version = "ICSDashboard"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: object) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _authorized(self) -> bool:
            """True when no password is required or the Basic credentials match."""
            if not password:
                return True
            client = _client_id(self)
            with lock:
                if len(_recent_failures(client)) >= MAX_FAILURES:
                    self._json(429, {"error": "محاولات كثيرة. حاول لاحقًا."})
                    return False
            supplied = ""
            header = self.headers.get("Authorization", "")
            if header.startswith("Basic "):
                try:
                    decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
                    supplied = decoded.partition(":")[2]  # the user name is ignored
                except Exception:
                    supplied = ""
            if supplied and hmac.compare_digest(supplied.encode("utf-8"), password.encode("utf-8")):
                return True
            if supplied:
                with lock:
                    _recent_failures(client).append(time.monotonic())
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="ICS", charset="UTF-8"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            path = self.path.split("?", 1)[0]
            if path == "/healthz":  # for the platform's port/health check; no data
                self._send(200, b"ok", "text/plain; charset=utf-8")
                return
            if locked:
                self._json(503, {"error": "اللوحة مقفلة: اضبط DASHBOARD_PASSWORD."})
                return
            if not self._authorized():
                return
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path.startswith("/api/"):
                name = path[len("/api/"):]
                fn: Optional[Callable[[], object]] = routes.get(name)
                if fn is None:
                    self._json(404, {"error": "unknown endpoint"})
                    return
                try:
                    self._json(200, fn())
                except Exception:  # never leak a stack trace to the browser
                    log.exception("dashboard endpoint %s failed", name)
                    self._json(500, {"error": "تعذّر تحميل البيانات"})
                return
            self._json(404, {"error": "not found"})

        def _reject(self) -> None:
            self.send_response(405)
            self.send_header("Allow", "GET")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_POST = do_PUT = do_DELETE = do_PATCH = _reject  # noqa: N815

        def log_message(self, fmt: str, *args) -> None:
            log.debug("web: " + fmt, *args)

    return Handler


def create_server(
    config: Config,
    host: str = "127.0.0.1",
    port: int = 8765,
    password: Optional[str] = None,
    locked: bool = False,
) -> ThreadingHTTPServer:
    if host not in LOOPBACK_HOSTS and not (password or locked):
        raise ValueError(
            f"Refusing to bind to {host!r} without a password: the dashboard shows the portfolio."
        )
    return ThreadingHTTPServer((host, port), make_handler(config, password, locked))


def start_hosted(config: Config) -> Optional[threading.Thread]:
    """Run the dashboard inside the bot's process when the platform gives a PORT.

    Returns ``None`` when there is no PORT (a plain worker / local run). With a
    PORT but no ``DASHBOARD_PASSWORD`` it starts *locked* — health check only —
    so the platform's port scan passes and nothing is exposed.
    """
    raw = os.environ.get("PORT", "").strip()
    if not raw:
        return None
    password = (config.env.dashboard_password or "").strip() or None
    server = create_server(config, host="0.0.0.0", port=int(raw), password=password, locked=password is None)
    if password is None:
        log.warning("PORT is set but DASHBOARD_PASSWORD is not: dashboard stays locked.")
    else:
        log.info("ICS dashboard on port %s (read-only, password-protected).", raw)
    thread = threading.Thread(target=server.serve_forever, name="ics-dashboard", daemon=True)
    thread.start()
    return thread


def serve(config: Config, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = create_server(config, host, port)
    log.info("ICS dashboard: http://%s:%d  (read-only, Ctrl+C to stop)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

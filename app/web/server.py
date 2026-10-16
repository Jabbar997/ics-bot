"""Read-only local web dashboard for ICS.

A second front end next to the Telegram bot, built on the same
:class:`CommandService`, so the dashboard and the bot can never disagree.

Deliberately narrow:
* **Read-only.** Only the report commands in :data:`READ_COMMANDS` are served.
  ``stop`` / ``resume`` are not routable, and only GET is accepted.
* **Local only.** Binds to loopback; :func:`serve` refuses any other host.
* **No new dependencies.** Standard library only.
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from app.config import Config
from app.db.database import session_scope
from app.db.repositories import PositionRepository
from app.risk.kill_switch import KillSwitchManager
from app.telegram.commands import SAFETY, CommandService

log = logging.getLogger(__name__)

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
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


def make_handler(config: Config) -> type:
    service = CommandService(config)
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

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            path = self.path.split("?", 1)[0]
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


def create_server(config: Config, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"The dashboard is read-only and local-only; refusing to bind to {host!r}."
        )
    return ThreadingHTTPServer((host, port), make_handler(config))


def serve(config: Config, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = create_server(config, host, port)
    log.info("ICS dashboard: http://%s:%d  (read-only, Ctrl+C to stop)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

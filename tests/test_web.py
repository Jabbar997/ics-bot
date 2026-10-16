"""The local dashboard: read-only, loopback-only, and consistent with the bot."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from app.telegram.commands import ALL_COMMANDS, CommandService
from app.web.server import READ_COMMANDS, create_server


@pytest.fixture
def base_url(config, db_url):
    server = create_server(config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers, r.read().decode("utf-8")


def test_refuses_to_bind_to_a_public_interface(config):
    with pytest.raises(ValueError):
        create_server(config, host="0.0.0.0", port=0)


def test_control_commands_are_not_exposed():
    assert not {"stop", "resume"} & set(READ_COMMANDS)


def test_every_read_command_exists_on_the_bot_service():
    assert set(READ_COMMANDS) <= set(ALL_COMMANDS)


def test_index_is_served_rtl(base_url):
    status, headers, body = _get(base_url + "/")
    assert status == 200 and 'dir="rtl"' in body
    assert "text/html" in headers["Content-Type"]


def test_summary_has_the_card_numbers(base_url, config):
    _, _, body = _get(base_url + "/api/summary")
    data = json.loads(body)
    assert data["cash"] == pytest.approx(config.initial_capital)
    assert data["positions"] == [] and data["frozen"] is False


def test_report_matches_what_the_bot_would_send(base_url, config):
    _, _, body = _get(base_url + "/api/portfolio")
    assert json.loads(body)["text"] == CommandService(config).portfolio()


@pytest.mark.parametrize("name", ["stop", "resume", "nope"])
def test_unknown_or_control_endpoints_404(base_url, name):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{base_url}/api/{name}")
    assert e.value.code == 404


def test_writes_are_rejected_and_do_not_change_state(base_url, config):
    req = urllib.request.Request(base_url + "/api/stop", method="POST", data=b"")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 405
    assert json.loads(_get(base_url + "/api/summary")[2])["frozen"] is False


# --------------------------- hosted / password mode -------------------------- #
import base64  # noqa: E402

from app.web.server import MAX_FAILURES, start_hosted  # noqa: E402


def _basic(password, user="x"):
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def _req(url, password=None, headers=None):
    h = dict(headers or {})
    if password is not None:
        h["Authorization"] = _basic(password)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h)) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, ""


def _serve(config, **kw):
    server = create_server(config, port=0, **kw)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_public_bind_needs_a_password_or_lock(config):
    with pytest.raises(ValueError):
        create_server(config, host="0.0.0.0", port=0)
    create_server(config, host="0.0.0.0", port=0, password="s3cret").server_close()
    create_server(config, host="0.0.0.0", port=0, locked=True).server_close()


def test_password_is_required_and_checked(config, db_url):
    server, base = _serve(config, password="s3cret")
    try:
        assert _req(base + "/api/summary")[0] == 401
        assert _req(base + "/api/summary", "wrong")[0] == 401
        assert _req(base + "/api/summary", "s3cret")[0] == 200
        assert _req(base + "/", "s3cret")[0] == 200
        # an empty password must never match
        assert _req(base + "/api/summary", "")[0] == 401
    finally:
        server.shutdown(); server.server_close()


def test_health_check_is_public_and_carries_no_data(config, db_url):
    server, base = _serve(config, password="s3cret")
    try:
        status, body = _req(base + "/healthz")
        assert status == 200 and body == "ok"
    finally:
        server.shutdown(); server.server_close()


def test_repeated_wrong_passwords_lock_the_client_out(config, db_url):
    server, base = _serve(config, password="s3cret")
    try:
        for _ in range(MAX_FAILURES):
            assert _req(base + "/api/summary", "wrong")[0] == 401
        assert _req(base + "/api/summary", "wrong")[0] == 429
        assert _req(base + "/api/summary", "s3cret")[0] == 429  # even the right one, while locked
    finally:
        server.shutdown(); server.server_close()


def test_lockout_cannot_be_dodged_by_spoofing_forwarded_for(config, db_url):
    server, base = _serve(config, password="s3cret")
    try:
        for i in range(MAX_FAILURES):
            _req(base + "/api/summary", "wrong", {"X-Forwarded-For": f"9.9.9.{i}, 5.5.5.5"})
        # the proxy-appended (last) entry is what counts, so this is still locked out
        assert _req(base + "/api/summary", "wrong", {"X-Forwarded-For": "1.2.3.4, 5.5.5.5"})[0] == 429
    finally:
        server.shutdown(); server.server_close()


def test_locked_mode_serves_health_only(config, db_url):
    server, base = _serve(config, locked=True)
    try:
        assert _req(base + "/healthz")[0] == 200
        assert _req(base + "/api/summary")[0] == 503
        assert _req(base + "/")[0] == 503
    finally:
        server.shutdown(); server.server_close()


def test_start_hosted_is_a_noop_without_port(config, monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert start_hosted(config) is None


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_start_hosted_with_password_serves_the_dashboard(config, db_url, monkeypatch):
    port = _free_port()
    monkeypatch.setenv("PORT", str(port))
    monkeypatch.setattr(config.env, "dashboard_password", "s3cret")
    assert start_hosted(config) is not None
    base = f"http://127.0.0.1:{port}"
    assert _req(base + "/api/summary")[0] == 401
    assert _req(base + "/api/summary", "s3cret")[0] == 200


def test_start_hosted_without_password_stays_locked(config, db_url, monkeypatch):
    port = _free_port()
    monkeypatch.setenv("PORT", str(port))
    monkeypatch.setattr(config.env, "dashboard_password", "")
    assert start_hosted(config) is not None
    base = f"http://127.0.0.1:{port}"
    assert _req(base + "/healthz")[0] == 200   # the platform's port check passes
    assert _req(base + "/api/summary")[0] == 503

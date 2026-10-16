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

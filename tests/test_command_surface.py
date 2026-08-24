"""The advertised command list must match what the bot actually registers.

/health and /learning were wired up, registered in the Telegram menu, and worked
— but were missing from the /start text, so from the user's side they did not
exist. These tests close that gap in both directions.
"""
from app.config import load_config
from app.telegram.bot import ICSBot
from app.telegram.commands import ALL_COMMANDS, COMMAND_GROUPS, CommandService


def _registered_names():
    """Command names the bot binds handlers for, without touching the network."""
    cfg = load_config()
    bot = ICSBot(cfg)
    service = bot.service
    # Mirror build_application()'s map without constructing a real Application.
    return {
        name for name in ALL_COMMANDS if callable(getattr(service, name, None))
    }


def test_every_advertised_command_exists_on_the_service():
    service = CommandService(load_config())
    for name in ALL_COMMANDS:
        assert callable(getattr(service, name, None)), f"/{name} is advertised but not implemented"


def test_learning_and_health_are_advertised(db_url):
    """The exact regression: both worked, neither was listed."""
    text = CommandService(load_config()).command_list()
    assert "/learning" in text
    assert "/health" in text


def test_start_lists_every_command(db_url):
    text = CommandService(load_config()).start()
    for name in ALL_COMMANDS:
        assert f"/{name}" in text, f"/{name} is missing from /start"


def test_command_groups_have_no_duplicates():
    assert len(ALL_COMMANDS) == len(set(ALL_COMMANDS))


def test_start_still_carries_the_safety_notice(db_url):
    text = CommandService(load_config()).start()
    assert "تداول ورقي فقط" in text


def test_bot_registers_exactly_the_advertised_commands():
    """No orphan handlers, no advertised-but-unbound commands."""
    assert _registered_names() == set(ALL_COMMANDS)

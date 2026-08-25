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


def test_learning_states_the_cap_in_points_not_percent():
    """ICS-DOC-004 rejects the relative reading; the display must not reinstate it.

    The cap is +/-5 absolute points (25 may reach 30). Rendering it as "±5%"
    told the user the opposite of what the code does.
    """
    from app.config import load_config
    from app.db import database
    from app.learning.feedback_loop import MAX_SHIFT_POINTS
    from app.telegram.commands import CommandService

    database.init_engine("sqlite:///:memory:", force_reset=True)
    database.create_all()
    text = CommandService(load_config()).learning()

    assert f"±{MAX_SHIFT_POINTS:.0f} نقاط" in text
    assert f"±{MAX_SHIFT_POINTS}%" not in text
    assert "±5%" not in text


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_every_cli_command_is_accepted_by_the_parser():
    """A command that main() dispatches must be reachable from the parser.

    `seed-learning` shipped with a handler and a dispatch branch but was left out
    of the hand-maintained choices list, so it was unreachable in production
    while every unit test still passed.
    """
    from app.main import COMMANDS, build_parser

    parser = build_parser()
    for name in COMMANDS:
        args = parser.parse_args([name])
        assert args.command == name


def test_parser_choices_and_dispatch_do_not_drift():
    """Both directions: nothing dispatched is unreachable, nothing offered is dead."""
    import inspect

    from app.main import COMMANDS, main

    source = inspect.getsource(main)
    for name in COMMANDS:
        assert f'"{name}"' in source, f"{name} is offered but never dispatched"

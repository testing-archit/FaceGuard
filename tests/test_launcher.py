import socket

import launcher


def test_free_port_skips_busy_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        port = launcher.free_port(busy)
        assert port != busy and port > 0


def test_app_command_is_local_only():
    cmd = launcher.app_command(launcher.venv_python(), 8765)
    assert cmd[cmd.index("--server.address") + 1] == "localhost"
    assert cmd[cmd.index("--browser.gatherUsageStats") + 1] == "false"
    assert cmd[-1] == "false" and str(launcher.APP) in cmd

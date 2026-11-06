"""TESTS-REACH-THE-NETWORK-1 (2026-11-05): no test reaches the network.

MEASURED 2026-11-05 on the whole suite (an audit hook in every Python process, joined to the HTTPS proxy's refusal log): 15 tests
reached rest.uniprot.org, ClinVar's FTP server, storage.googleapis.com, databases.lovd.nl, alphafold.ebi.ac.uk and api.github.com,
and collection started R five times in the checkout, where renv tried to bootstrap itself from bioconductor.org and
packagemanager.posit.co. Each failure was swallowed by the code that made it, so the tests passed -- in the sandbox, where every
connection was refused; on a machine with the network open the calls were made and their answers reached the tests.

The repository root's conftest.py now sets the three offline variables for the session and refuses every non-loopback name lookup,
connection and datagram, and every connection to a configured proxy, inside the call; a test that tried fails at teardown, an
attempt while the session starts or tests are collected stops the session. Pinned here, in this session and in child sessions
running the root conftest.py (child_conftest):
* this session is guarded: the guard is armed, the offline variables are 1 and huggingface_hub read them at its import;
* what counts as the loopback; which proxy settings are refused (each one on the loopback, its port defaulted by its scheme, its
  credentials not kept), read from every <scheme>_proxy variable AND the operating system's own settings (GUARD-SYSTEM-PROXIES-1);
* a loopback server is untouched here;
* in a child session: a refused connection raises NetworkRefused and fails its test at teardown, naming the target and the test's
  own line; a refusal the code swallowed fails all the same; a name lookup is refused; a connection to the configured proxy is
  refused while another loopback port is not; an xfail-marked test that tried fails the session; a test that made no attempt
  passes; the summary counts and names them, and names the proxy refused; the invoker's own offline value is replaced and named;
* in a child session: a module that tries at import time stops the session before any test runs, naming the attempt.

Author: Monzia Moodie
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROOT_CONFTEST = REPO / "conftest.py"

# --------------------------------------------------------------------------- in this session


def _root_conftest_module(config):
    found = [plugin for plugin in config.pluginmanager.get_plugins()
             if Path(getattr(plugin, "__file__", None) or "-").resolve() == ROOT_CONFTEST]
    assert len(found) == 1, found
    return found[0]


def test_this_session_is_guarded_and_offline(request):
    guard = _root_conftest_module(request.config)
    assert guard._NETWORK["armed"] is True
    for name in guard._OFFLINE_VARIABLES:
        assert os.environ[name] == "1", name
    from huggingface_hub import constants

    assert constants.HF_HUB_OFFLINE is True            # read at huggingface_hub's own import: the variable was set before it


@pytest.mark.parametrize("host, loopback", [
    ("127.0.0.1", True), ("127.8.8.8", True), ("::1", True), ("[::1]", True), ("localhost", True), ("LOCALHOST", True),
    ("0.0.0.0", True), ("::", True), (None, True), ("", True), (b"127.0.0.1", True),
    ("192.0.2.1", False), ("10.0.0.1", False), ("example.org", False), ("fe80::1%eth0", False), ("2001:db8::1", False)],
    ids=["ipv4-loopback", "ipv4-loopback-net", "ipv6-loopback", "ipv6-loopback-bracketed", "localhost", "localhost-upper",
         "ipv4-unspecified", "ipv6-unspecified", "none-bind", "empty-bind", "bytes-loopback", "ipv4-documentation", "ipv4-private",
         "a-name", "ipv6-link-local-scoped", "ipv6-documentation"])
def test_what_counts_as_the_loopback(request, host, loopback):
    assert _root_conftest_module(request.config)._is_loopback(host) is loopback


@pytest.mark.parametrize("settings, expected", [
    ({"https": "http://127.0.0.1:39533"}, {("127.0.0.1", 39533)}),
    ({"http": "localhost:3128"}, {("localhost", 3128)}),
    ({"https": "http://user:secret@127.0.0.1:8888"}, {("127.0.0.1", 8888)}),
    ({"http": "http://127.0.0.1"}, {("127.0.0.1", 80)}),
    ({"https": "https://127.0.0.1"}, {("127.0.0.1", 443)}),
    ({"socks": "socks5://[::1]"}, {("::1", 1080)}),
    ({"https": "http://proxy.example:8080", "all": "socks5://10.1.2.3:1080"}, set()),
    ({"no": "localhost"}, set()),                       # no_proxy names hosts; read as a URL it would be localhost:80
    ({"http": "http://[bad"}, set()),
    ({"http": "http://127.0.0.1:not-a-port"}, set()),
    ({"claude_code_websearch_use_ccr": "1"}, set())],
    ids=["https", "bare", "credentials-dropped", "http-default-port", "https-default-port", "socks-default-port-ipv6",
         "elsewhere-refused-as-any-address", "no-proxy-is-not-a-proxy", "unparsable", "bad-port", "not-a-proxy-variable"])
def test_which_proxy_settings_are_refused(request, settings, expected):
    assert _root_conftest_module(request.config)._loopback_proxies(settings) == frozenset(expected)


def test_the_proxies_are_read_from_every_variable_and_the_system_settings(request, monkeypatch):
    import urllib.request

    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("ftp_proxy", "http://127.0.0.1:2121")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:39533")
    monkeypatch.delattr(urllib.request, "getproxies_macosx_sysconf", raising=False)
    monkeypatch.setattr(urllib.request, "getproxies_registry",
                        lambda: {"http": "http://127.0.0.1:3128", "https": "http://127.0.0.1:39533"}, raising=False)
    assert _root_conftest_module(request.config)._configured_proxies() == {
        ("127.0.0.1", 2121): "the environment", ("127.0.0.1", 3128): "the Windows Internet settings",
        ("127.0.0.1", 39533): "the environment and the Windows Internet settings"}


def test_a_loopback_server_is_untouched(request):
    guard = _root_conftest_module(request.config)
    before = len(guard._NETWORK["attempts"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=5) as client:
            peer, _ = server.accept()
            with peer:
                client.sendall(b"ok")
                assert peer.recv(2) == b"ok"
    assert len(guard._NETWORK["attempts"]) == before


# --------------------------------------------------------------------------- child sessions

CHILD_INI = "[pytest]\n"

REACHING = '''
import socket

import pytest


def test_a_refused_connection():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        with pytest.raises(OSError) as refused:
            s.connect(("192.0.2.1", 443))
    finally:
        s.close()
    assert type(refused.value).__name__ == "NetworkRefused" and isinstance(refused.value, ConnectionRefusedError)


def test_a_swallowed_lookup():
    try:
        socket.getaddrinfo("gvc-planted.example", 443)
    except OSError:
        pass


def test_the_configured_proxy_is_refused():
    import os
    port = int(os.environ["HTTPS_PROXY"].rsplit(":", 1)[1])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", port))
    except OSError:
        pass
    finally:
        s.close()


def test_another_loopback_port_is_untouched():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=5):
            server.accept()[0].close()


@pytest.mark.xfail(reason="planted: the xfail must not hide the attempt", strict=False)
def test_an_xfail_marked_attempt():
    try:
        socket.getaddrinfo("gvc-planted-xfail.example", 80)
    except OSError:
        pass
    assert False


def test_no_attempt_and_the_offline_variables():
    import os
    assert [os.environ[n] for n in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")] == ["1", "1", "1"]
'''

XFAIL_ALONE = '''
import socket

import pytest


@pytest.mark.xfail(reason="planted: the only finding of the session is hidden as an xfail", strict=False)
def test_an_xfail_marked_attempt():
    try:
        socket.getaddrinfo("gvc-planted-xfail.example", 80)
    except OSError:
        pass
    assert False


def test_quiet():
    pass
'''

AT_IMPORT = '''
import socket

try:
    socket.getaddrinfo("gvc-planted-import.example", 443)
except OSError:
    pass


def test_never_runs():
    pass
'''


def _child(tmp_path: Path, environment, conftest, tests: str, **extra) -> tuple:
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "pytest.ini").write_text(CHILD_INI, encoding="utf-8")
    (project / "conftest.py").write_bytes(conftest())
    (project / "tests" / "test_child.py").write_text(tests, encoding="utf-8")
    env = environment(PYTHONPATH=os.pathsep.join([str(REPO / "src")] +
                                                 ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])), **extra)
    for name in [name for name in env if name.lower().endswith("_proxy")]:
        env.pop(name)
    env["HTTPS_PROXY"] = "http://127.0.0.1:47"
    run = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-rA", "--confcutdir",
                          str(project), "--junitxml", str(tmp_path / "child.xml")],
                         cwd=str(project), env=env, capture_output=True, text=True, timeout=600)
    return run.returncode, run.stdout + run.stderr, tmp_path / "child.xml"


def _records(junit: Path) -> dict:
    """{test name: [(element, its type or message start), ...]} -- what junit recorded for each test, exactly."""
    return {case.get("name"): [(k.tag, k.get("type") if k.tag == "skipped" else (k.get("message") or "")[:60]) for k in case]
            for case in ET.parse(junit).getroot().iter("testcase")}


def test_every_attempt_is_refused_and_fails_its_test_and_the_session(tmp_path, child_pytest_environment, child_conftest):
    code, out, junit = _child(tmp_path, child_pytest_environment, child_conftest, REACHING, HF_HUB_OFFLINE="0")
    assert code == 1, out
    records = _records(junit)
    refused = [("error", 'failed on teardown with "AssertionError: THIS TEST TRIED TO ')]
    assert records["test_a_refused_connection"] == refused, (records, out)        # its call passed; its teardown failed it
    assert records["test_a_swallowed_lookup"] == refused, (records, out)
    assert records["test_the_configured_proxy_is_refused"] == refused, (records, out)
    assert records["test_another_loopback_port_is_untouched"] == [], (records, out)
    assert records["test_no_attempt_and_the_offline_variables"] == [], (records, out)
    # pytest records the teardown finding of an xfail-marked test as a second xfail -- hidden there; the session fails regardless
    assert records["test_an_xfail_marked_attempt"] == [("skipped", "pytest.xfail")] * 2, (records, out)
    assert "4 passed" not in out and re.search(r"\b5 passed, 2 xfailed, 3 errors\b", out), out
    # each finding names the event, the target and the test's own line
    assert re.search(r"socket\.connect 192\.0\.2\.1:443 -- during call of tests/test_child\.py::test_a_refused_connection -- at "
                     r"tests[/\\]test_child\.py:\d+ test_a_refused_connection", out), out
    assert re.search(r"socket\.getaddrinfo gvc-planted\.example:443 -- during call of tests/test_child\.py::test_a_swallowed_lookup",
                     out), out
    assert re.search(r"socket\.connect 127\.0\.0\.1:47 \(a configured proxy\) -- during call of "
                     r"tests/test_child\.py::test_the_configured_proxy_is_refused", out), out
    assert "THIS TEST TRIED TO REACH THE NETWORK (TESTS-REACH-THE-NETWORK-1): 1 attempt, each refused" in out, out
    # the summary counts and names them, the xfail-marked one included, and names the replaced offline value
    section = out[out.index("network guard"):]
    assert ("6 tests checked, 4 tried to reach the network; 4 attempts refused in all; set for the session: HF_HUB_OFFLINE=1, "
            "TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1; replaced: HF_HUB_OFFLINE (the invoker's '0')") in section, section
    proxies = [line for line in section.splitlines() if line.startswith("proxies on the loopback, refused as the network: ")]
    assert len(proxies) == 1 and "127.0.0.1:47 (the environment)" in proxies[0], section
    named = section[section.index("THESE TESTS TRIED TO REACH THE NETWORK"):].splitlines()[1:5]
    assert [line.strip() for line in named] == [
        "tests/test_child.py::test_a_refused_connection", "tests/test_child.py::test_a_swallowed_lookup",
        "tests/test_child.py::test_the_configured_proxy_is_refused", "tests/test_child.py::test_an_xfail_marked_attempt"], section


def test_an_attempt_at_import_stops_the_session_before_any_test(tmp_path, child_pytest_environment, child_conftest):
    code, out, _ = _child(tmp_path, child_pytest_environment, child_conftest, AT_IMPORT)
    assert code == pytest.ExitCode.USAGE_ERROR, out
    assert "CODE RUN WHILE THE SESSION STARTED OR TESTS WERE COLLECTED TRIED TO REACH THE NETWORK (TESTS-REACH-THE-NETWORK-1)" in out
    assert re.search(r"socket\.getaddrinfo gvc-planted-import\.example:443 -- during collection of \(no test\) -- at "
                     r"tests[/\\]test_child\.py:\d+ <module>", out), out
    assert "PASSED" not in out and " passed" not in out, out                           # stopped before any test ran


def test_an_attempt_hidden_as_an_xfail_still_fails_the_session(tmp_path, child_pytest_environment, child_conftest):
    """The xfail-marked attempt is the session's ONLY finding: pytest reports its teardown failure as a second xfail, so without the
    guard's own exit status the run would pass."""
    code, out, junit = _child(tmp_path, child_pytest_environment, child_conftest, XFAIL_ALONE)
    assert re.search(r"\b1 passed, 2 xfailed\b", out) and "error" not in out.split("network guard")[0].lower(), out
    assert code == 1, out
    section = out[out.index("network guard"):]
    assert "2 tests checked, 1 tried to reach the network; 1 attempt refused in all" in section, section
    assert "    tests/test_child.py::test_an_xfail_marked_attempt" in section, section

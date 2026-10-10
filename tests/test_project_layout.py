"""New-user entrypoint, asset location, export, and real CPU HTTP regression checks."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from jev_teaching import export_demo
from jev_teaching.paths import ARTIFACTS, REPO_ROOT, WEB
from jev_teaching.serve_demo import Demo
from jev_teaching.sokoban_env import ACTIONS, Layout, State, is_solved, step


def module_environment():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def cli_export(*arguments):
    with patch.object(sys, "argv", ["export_demo", *map(str, arguments)]), \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        export_demo.main()


def training_fixture(directory, browser=False):
    directory.mkdir()
    filenames = ("heads.pt", "report.json") if browser else ("policy.pt", "levels.json", "report.json")
    for name in filenames:
        (directory / name).write_bytes(("fixture:" + name).encode())
    return directory


class ProjectLayoutTests(unittest.TestCase):
    def test_browser_model_name_comes_from_loaded_report(self):
        demo = Demo(ARTIFACTS)
        demo.browser_report = dict(demo.browser_report, model="custom-backbone")
        information = demo.info()
        self.assertIn("custom-backbone", information["browser_model"])
        self.assertNotIn("Qwen3.5-0.8B", information["browser_model"])
        self.assertIsNone(demo.encoder)

    def test_documentation_local_links_resolve(self):
        documents = [REPO_ROOT / "README.md", ARTIFACTS / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
        missing, checked = [], 0
        for document in documents:
            self.assertTrue(document.is_file(), str(document))
            text = document.read_text(encoding="utf-8")
            # Code examples may intentionally show placeholder Markdown links.
            text = re.sub(r"(?ms)^\s*```[^\n]*\n.*?^\s*```\s*$", "", text)
            inline = re.findall(r"!?\[[^\]]*\]\(\s*(<[^>]*>|[^\s)]+)(?:\s+[^)]*)?\)", text)
            references = re.findall(r"(?m)^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)", text)
            for target in inline + references:
                target = target.strip("<>")
                parts = urlsplit(target)
                if parts.scheme in ("http", "https", "mailto") or not parts.path:
                    continue
                relative = unquote(parts.path)
                resolved = ((REPO_ROOT / relative.lstrip("/")) if relative.startswith("/")
                            else (document.parent / relative)).resolve()
                checked += 1
                if not resolved.exists():
                    missing.append(f"{document.relative_to(REPO_ROOT)} -> {target}")
        self.assertGreater(checked, 0, "No local documentation links were checked")
        self.assertEqual(missing, [], "Broken local Markdown links:\n" + "\n".join(missing))

    def test_assets_are_anchored_to_package_not_current_directory(self):
        self.assertEqual(REPO_ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(ARTIFACTS, REPO_ROOT / "artifacts")
        self.assertEqual(WEB, REPO_ROOT / "jev_teaching" / "web")
        self.assertTrue((WEB / "index.html").is_file())
        for name in ("sokoban-demo.pt", "sokoban-levels.json", "sokoban-results.json"):
            self.assertTrue((ARTIFACTS / "sokoban" / name).is_file(), name)

    def test_module_help_from_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "jev_teaching", "--help"],
                                    cwd=directory, env=module_environment(), capture_output=True,
                                    text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--artifacts", result.stdout)
        self.assertIn("--port", result.stdout)

    def test_export_default_destination_is_bundled_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            source = training_fixture(Path(directory) / "run")
            # Inspect the default mapping without changing checked-in artifacts.
            with patch.object(export_demo.shutil, "copy2") as copying, \
                    patch.object(Path, "mkdir"):
                cli_export("--sokoban-run", source, "--overwrite")
            destinations = {Path(call.args[1]) for call in copying.call_args_list}
        self.assertEqual(destinations, {ARTIFACTS / "sokoban" / name for name in
                         ("sokoban-demo.pt", "sokoban-levels.json", "sokoban-results.json")})

    def test_export_custom_destination_and_atomic_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            game = training_fixture(base / "game-run")
            browser = training_fixture(base / "browser-run", browser=True)
            destination = base / "new-artifacts"
            cli_export("--sokoban-run", game, "--browser-run", browser, "--dest", destination)
            expected = {"sokoban/sokoban-demo.pt": "policy.pt", "sokoban/sokoban-levels.json": "levels.json",
                        "sokoban/sokoban-results.json": "report.json", "browser/browser-demo.pt": "heads.pt",
                        "browser/browser-results.json": "report.json"}
            for target, original in expected.items():
                self.assertEqual((destination / target).read_bytes(), ("fixture:" + original).encode())
            original_bytes = {name: (destination / name).read_bytes() for name in expected}
            (game / "policy.pt").write_bytes(b"changed source must not overwrite silently")
            with self.assertRaises(SystemExit) as failure:
                cli_export("--sokoban-run", game, "--dest", destination)
            self.assertEqual(failure.exception.code, 2)
            self.assertEqual({name: (destination / name).read_bytes() for name in expected}, original_bytes)
            cli_export("--sokoban-run", game, "--dest", destination, "--overwrite")
            self.assertEqual((destination / "sokoban" / "sokoban-demo.pt").read_bytes(),
                             b"changed source must not overwrite silently")

    def test_export_browser_head_can_include_bundled_game(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            browser = training_fixture(base / "browser-run", browser=True)
            destination = base / "combined"
            cli_export("--browser-run", browser, "--include-bundled-sokoban", "--dest", destination)
            for name in ("sokoban-demo.pt", "sokoban-levels.json", "sokoban-results.json"):
                self.assertEqual((destination / "sokoban" / name).read_bytes(),
                                 (ARTIFACTS / "sokoban" / name).read_bytes())
            self.assertEqual((destination / "browser" / "browser-demo.pt").read_bytes(), b"fixture:heads.pt")

    def test_actual_http_with_custom_artifacts_without_transformers(self):
        # Block optional transformers imports in the *new* interpreter. This
        # checks the advertised standalone game path without downloading Qwen.
        bootstrap = """
import importlib.abc
import runpy
import sys
class NoTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'transformers' or fullname.startswith('transformers.'):
            raise ModuleNotFoundError('transformers intentionally unavailable in game-only test')
        return None
sys.meta_path.insert(0, NoTransformers())
runpy.run_module('jev_teaching', run_name='__main__')
"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            custom = base / "custom-artifacts"
            game = custom / "sokoban"
            game.mkdir(parents=True)
            for name in ("sokoban-demo.pt", "sokoban-results.json"):
                shutil.copy2(ARTIFACTS / "sokoban" / name, game / name)
            all_levels = json.loads((ARTIFACTS / "sokoban" / "sokoban-levels.json").read_text(encoding="utf-8"))
            chosen = next(record for record in all_levels if record["split"] == "test")
            (game / "sokoban-levels.json").write_text(json.dumps([chosen]), encoding="utf-8")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            process = subprocess.Popen(
                [sys.executable, "-c", bootstrap, "--artifacts", str(custom), "--port", str(port), "--device", "cpu"],
                cwd=base, env=module_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            address = f"http://127.0.0.1:{port}"
            try:
                info = None
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        output, errors = process.communicate(timeout=5)
                        self.fail(f"Server exited before readiness:\n{output}\n{errors}")
                    try:
                        with urlopen(address + "/api/info", timeout=1) as response:
                            info = json.load(response)
                        break
                    except (OSError, URLError):
                        time.sleep(.1)
                self.assertIsNotNone(info, "Server did not become ready within 30 seconds")
                self.assertTrue(info["ready"])
                self.assertFalse(info["browser_available"])
                with urlopen(address, timeout=5) as response:
                    self.assertIn(b"<html", response.read().lower())
                with urlopen(address + "/api/levels", timeout=5) as response:
                    levels = json.load(response)["levels"]
                self.assertEqual([level["id"] for level in levels], [chosen["id"]])
                level = levels[0]
                payload = {"level_id": level["id"], "player": level["player"], "boxes": level["boxes"], "policy": "after"}
                request = Request(address + "/api/step", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=10) as response:
                    result = json.load(response)
                layout = Layout(chosen["width"], chosen["height"], frozenset(chosen["walls"]), tuple(chosen["goals"]))
                initial = State(chosen["player"], tuple(chosen["boxes"]))
                following, moved = step(layout, initial, ACTIONS.index(result["action"]))
                xy = lambda cell: [cell % layout.width, cell // layout.width]
                self.assertTrue(moved)
                self.assertEqual(result["next_state"], {"player": xy(following.player), "boxes": [xy(box) for box in following.boxes]})
                self.assertEqual(result["solved"], is_solved(layout, following))
                self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=5)
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()

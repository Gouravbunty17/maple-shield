from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_launchers_are_documented():
    demo = (ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")

    assert "scripts\\demo_live.ps1" in demo or "scripts/demo_live.ps1" in demo
    assert "scripts/demo_live.sh" in demo
    assert "make demo" in demo


def test_dev_script_uses_builtin_mock_source():
    dev = (ROOT / "scripts" / "dev.sh").read_text(encoding="utf-8")

    assert "--source mock" in dev
    assert "--source samples/mock" not in dev


def test_demo_scripts_support_dry_run():
    ps1 = (ROOT / "scripts" / "demo_live.ps1").read_text(encoding="utf-8")
    sh = (ROOT / "scripts" / "demo_live.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "DryRun" in ps1
    assert "--dry-run" in sh
    assert "$(DEMO_ARGS)" in makefile

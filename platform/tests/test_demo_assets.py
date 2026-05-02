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


def test_smoke_runner_is_documented_and_make_wired():
    demo = (ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts" / "smoke_demo.py").read_text(encoding="utf-8")

    assert "python scripts/smoke_demo.py" in demo
    assert "smoke-demo" in makefile
    assert "scripts/smoke_demo.py" in makefile
    assert "audit_verified" in smoke


def test_browser_demo_qa_is_documented():
    demo = (ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")
    qa = (ROOT / "docs" / "DEMO_QA.md").read_text(encoding="utf-8")

    assert "platform/docs/DEMO_QA.md" in demo
    assert "http://127.0.0.1:5173" in demo
    assert "Browser-visible demo QA passed" in qa
    assert "Audit chain" in qa

@echo off
title Maple Shield — Demo Launcher
color 0A

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║          MAPLE SHIELD  —  DEMO LAUNCHER         ║
echo  ║       Protecting Airspace With Intelligence      ║
echo  ╚══════════════════════════════════════════════════╝
echo.
echo  Starting all systems...
echo.

cd /d "%~dp0"

REM ── 1. Replay Dashboard (localhost:5000) ──────────────────────
echo  [1/3]  Starting Replay Dashboard    → http://localhost:5000
start "Maple Shield — Replay Dashboard" cmd /k "python maple_shield_dashboard.py --port 5000"
timeout /t 2 /nobreak >nul

REM ── 2. C2 Simulator (localhost:5001) ─────────────────────────
echo  [2/3]  Starting C2 Simulator        → http://localhost:5001
start "Maple Shield — C2 Simulator" cmd /k "python maple_shield_c2_sim.py"
timeout /t 2 /nobreak >nul

REM ── 3. Open both in browser ───────────────────────────────────
echo  [3/3]  Opening dashboards in browser...
timeout /t 2 /nobreak >nul
start http://localhost:5000
start http://localhost:5001

echo.
echo  ════════════════════════════════════════════════════
echo   SYSTEMS READY
echo  ════════════════════════════════════════════════════
echo.
echo   Replay Dashboard  →  http://localhost:5000
echo   C2 Simulator      →  http://localhost:5001
echo.
echo  ════════════════════════════════════════════════════
echo   DETECTION OPTIONS
echo  ════════════════════════════════════════════════════
echo.
echo   Option A — Live camera / video:
echo     python maple_shield_mvp.py                  (webcam)
echo     python maple_shield_mvp.py --source VIDEO   (video file)
echo.
echo   Option B — Synthetic scenario (no camera needed):
echo     python maple_shield_sim.py --scenario incursion
echo     python maple_shield_sim.py --scenario swarm
echo     python maple_shield_sim.py --scenario recon
echo     python maple_shield_sim.py --scenario standoff
echo.
echo  ════════════════════════════════════════════════════
echo   CoT UDP → 239.2.3.1:6969  (ATAK multicast)
echo   MQTT    → localhost:1883
echo  ════════════════════════════════════════════════════
echo.
pause

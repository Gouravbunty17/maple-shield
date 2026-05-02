param(
    [string]$Source = "mock",
    [ValidateSet("mock", "cairn-yolo")]
    [string]$Detector = "mock",
    [string]$YoloModel = $env:MAPLE_SHIELD_YOLO_MODEL,
    [string]$YoloClasses = $env:MAPLE_SHIELD_YOLO_CLASSES,
    [int]$Frames = 180,
    [double]$Fps = 10.0,
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $Root ".demo-logs"
$Processes = @()

function Resolve-DemoConfig {
    if ($Source -ne "mock" -and -not (Test-Path -LiteralPath $Source)) {
        Write-Warning "Source '$Source' was not found. Falling back to the mock feed."
        $script:Source = "mock"
    }
    if ($Detector -eq "cairn-yolo" -and [string]::IsNullOrWhiteSpace($YoloModel)) {
        Write-Warning "No YOLO model was provided. Falling back to the mock detector."
        $script:Detector = "mock"
    }
}

function Join-Args([string[]]$Parts) {
    return ($Parts | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join " "
}

function Edge-AgentCommand {
    $args = @(
        "python -m edge_agent.main",
        "--source `"$Source`"",
        "--detector $Detector",
        "--n-frames $Frames",
        "--fps $Fps",
        "--fusion http://localhost:8090"
    )
    if ($Detector -eq "cairn-yolo") {
        $args += "--yolo-model `"$YoloModel`""
        if (-not [string]::IsNullOrWhiteSpace($YoloClasses)) {
            $args += "--yolo-classes `"$YoloClasses`""
        }
    }
    return Join-Args $args
}

function Start-DemoProcess([string]$Name, [string]$WorkingDirectory, [string]$Command) {
    $out = Join-Path $LogDir "$Name.out.log"
    $err = Join-Path $LogDir "$Name.err.log"
    Write-Host "[$Name] $Command"
    $process = Start-Process `
        -FilePath "powershell" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $Command) `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err `
        -WindowStyle Hidden `
        -PassThru
    $script:Processes += $process
}

function Wait-Health([string]$Name, [string]$Url) {
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                Write-Host "[$Name] healthy"
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "$Name did not become healthy at $Url"
}

Resolve-DemoConfig
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$edgeCommand = Edge-AgentCommand

if ($DryRun) {
    Write-Host "[demo] root: $Root"
    Write-Host "[demo] command-api: python -m uvicorn app.main:app --port 8080"
    Write-Host "[demo] fusion-engine: python -m uvicorn fusion.main:app --port 8090"
    Write-Host "[demo] edge-agent: $edgeCommand"
    Write-Host "[demo] operator-ui: npm run dev -- --host 127.0.0.1"
    exit 0
}

try {
    $env:MAPLE_SHIELD_LAWFUL_USE_ACK = "true"
    Start-DemoProcess "command-api" (Join-Path $Root "command-api") "python -m uvicorn app.main:app --port 8080"
    Wait-Health "command-api" "http://localhost:8080/healthz"

    Start-DemoProcess "fusion-engine" $Root "python -m uvicorn fusion.main:app --port 8090"
    Wait-Health "fusion-engine" "http://localhost:8090/healthz"

    Start-DemoProcess "edge-agent" (Join-Path $Root "edge-agent") $edgeCommand
    Start-DemoProcess "operator-ui" (Join-Path $Root "operator-ui") "npm run dev -- --host 127.0.0.1"

    Write-Host "[demo] logs: $LogDir"
    Write-Host "[demo] UI: http://localhost:5173"
    if (-not $NoBrowser) {
        Start-Process "http://localhost:5173"
    }
    Write-Host "[demo] Press Ctrl+C to stop."
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host "[demo] stopping..."
    foreach ($process in $Processes) {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

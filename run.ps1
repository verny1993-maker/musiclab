<#
run.ps1 — run MusicLab/AudioLab pipeline steps OUTSIDE the Hermes chat.

Why: heavy steps (audio download/analysis, Docker) block Hermes' single-threaded
gateway when launched from the dashboard chat -> "gateway websocket closed (1011)".
Running them as detached background jobs here keeps Hermes responsive.

Usage (open a normal PowerShell/Terminal, NOT the Hermes chat):
  .\run.ps1 up                 # start Docker services (qdrant + music-analysis)
  .\run.ps1 status             # docker + running jobs + last log tails
  .\run.ps1 parse   -- --url "https://set79.com/..." --artist "DJ Name"
  .\run.ps1 audio   -- --set data/sets/set_123.json
  .\run.ps1 meta    -- --input data/library/tracks_all.json
  .\run.ps1 library -- --workers 1 --resume     # the big batch (hours) — runs detached
  .\run.ps1 cards   -- --all
  .\run.ps1 logs               # tail the newest job log
  .\run.ps1 stop-jobs          # stop background pipeline jobs (does NOT touch Hermes)

Everything after `--` is passed straight to the script.
#>
param(
  [Parameter(Mandatory=$true)][string]$cmd,
  [Parameter(ValueFromRemainingArguments=$true)]$rest
)
$ErrorActionPreference='Stop'
$root = 'C:\ARTEM DIRECTORIA\AudioLab'
$py   = Join-Path $root '.venv\Scripts\python.exe'
$logdir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

# strip a leading literal '--' that PowerShell keeps in $rest
$args = @($rest | Where-Object { $_ -ne '--' })

function Start-Job([string]$name,[string]$script,[string[]]$scriptArgs){
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $log = Join-Path $logdir "$name-$stamp.log"
  $allArgs = @($script) + $scriptArgs
  $p = Start-Process -FilePath $py -ArgumentList $allArgs -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
  Write-Host "[$name] started detached  PID=$($p.Id)"
  Write-Host "  log: $log"
  Write-Host "  follow: Get-Content `"$log`" -Wait -Tail 20"
}

switch ($cmd) {
  'up' {
    Write-Host "Starting Docker services..."
    Push-Location (Join-Path $root 'qdrant'); docker compose up -d; Pop-Location
    docker start music-analysis-music-analysis-1 2>$null
    docker ps --format "table {{.Names}}\t{{.Status}}"
  }
  'status' {
    Write-Host "=== Docker ==="; docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    Write-Host "`n=== Running python jobs (AudioLab .venv) ==="
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -match 'AudioLab' } |
      ForEach-Object { "PID $($_.ProcessId): $((($_.CommandLine) -replace '\s+',' '))" }
    Write-Host "`n=== Newest log ==="
    $last = Get-ChildItem $logdir -Filter '*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Desc | Select-Object -First 1
    if($last){ "$($last.FullName)`n"; Get-Content $last.FullName -Tail 15 }
  }
  'parse'   { Start-Job 'parse'   '01_parse.py'                 $args }
  'audio'   { Start-Job 'audio'   '02_enrich_audio.py'          $args }
  'meta'    { Start-Job 'meta'    'enrich_library.py'           $args }
  'library' { Start-Job 'library' '03_enrich_audio_library.py'  $args }
  'index'   { Start-Job 'index'   '03_index_library.py'         $args }
  'cards'   { Start-Job 'cards'   '05_build_cards.py'           $args }
  'logs' {
    $last = Get-ChildItem $logdir -Filter '*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Desc | Select-Object -First 1
    if($last){ "Tailing $($last.FullName) (Ctrl+C to stop)"; Get-Content $last.FullName -Wait -Tail 30 } else { "no logs yet" }
  }
  'stop-jobs' {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -match 'AudioLab' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped PID $($_.ProcessId)" }
  }
  default { Write-Host "Unknown command '$cmd'. See header of run.ps1 for usage." }
}

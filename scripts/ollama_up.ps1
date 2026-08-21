<#
.SYNOPSIS
  Bring Ollama up cleanly and report whether the GPU is actually fit to run a long job.

.DESCRIPTION
  Three things went wrong on this machine that this script exists to prevent recurring:

  1. Two servers competing for port 11434 (a manually started `ollama serve` plus the tray app).
     The loser logs bind errors forever and the winner may have lost GPU discovery, so inference
     silently falls back to CPU at roughly a tenth of the speed.
  2. OLLAMA_CONTEXT_LENGTH defaulting to 4096, which is smaller than this project's 3.2k-token
     system prompt plus reports plus output. The prompt is then truncated silently and the labels
     are garbage with no error anywhere.
  3. The GPU pinned at its 300 MHz idle clock by thermal throttling, which looks like a working
     setup and produces a six-day job.

.EXAMPLE
  .\scripts\ollama_up.ps1
  .\scripts\ollama_up.ps1 -Model qwen3:14b -Benchmark
#>
param(
    [string]$Model = "qwen3:14b",
    [int]$ContextLength = 8192,
    [switch]$Benchmark
)

$ErrorActionPreference = "Stop"
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) { throw "Ollama not found. Install with: winget install --id Ollama.Ollama" }

Write-Host "`n[1/5] stopping any running Ollama" -ForegroundColor Cyan
# llama-server.exe is the child process that actually holds the VRAM. Killing only the `ollama`
# parent orphans it, still allocated, and every restart then leaks another one. Six of them had
# accumulated here and between them held all 16 GiB, after which new loads either fell back to CPU
# or crashed the server outright — with no error that pointed at the cause.
$stale = Get-Process -Name "ollama", "ollama app", "llama-server" -ErrorAction SilentlyContinue
if ($stale) {
    Write-Host ("      killing {0}: {1}" -f $stale.Count, (($stale | Group-Object Name |
        ForEach-Object { "$($_.Count)x $($_.Name)" }) -join ", "))
    $stale | Stop-Process -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 5

$leaked = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($leaked) {
    $leaked | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

$vramUsed = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n")[0].Trim()
Write-Host "      VRAM after cleanup: $vramUsed MiB"
if ($vramUsed -gt 1500) {
    Write-Host "      WARNING: $vramUsed MiB still held by something. Loading a 9 GiB model on top" -ForegroundColor Yellow
    Write-Host "               of this will OOM or silently fall back to CPU. Check:" -ForegroundColor Yellow
    Write-Host "               nvidia-smi --query-compute-apps=pid,process_name --format=csv" -ForegroundColor Yellow
}

Write-Host "[2/5] starting a single server (context $ContextLength)" -ForegroundColor Cyan
$env:OLLAMA_CONTEXT_LENGTH = "$ContextLength"
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "$ContextLength", "User")
$log = "$env:TEMP\ollama_serve.log"
Start-Process -FilePath $ollama -ArgumentList "serve" `
    -RedirectStandardError $log -RedirectStandardOutput "$env:TEMP\ollama_serve_out.log" -WindowStyle Hidden

$up = $false
foreach ($i in 1..30) {
    Start-Sleep -Seconds 2
    try { Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 5 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { throw "server did not come up; see $log" }
Write-Host "      up at http://localhost:11434" -ForegroundColor Green

Write-Host "[3/5] checking the model is registered" -ForegroundColor Cyan
# A half-pulled model leaves its blobs on disk but no manifest, which looks like a finished
# download until the first request 404s. `ollama list` is the authority, not the blob directory.
$tags = (Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 15).models.name
if ($tags -notcontains $Model) {
    Write-Host "      $Model not registered - pulling (resumes any partial download)" -ForegroundColor Yellow
    & $ollama pull $Model
    $tags = (Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 15).models.name
    if ($tags -notcontains $Model) { throw "pull did not complete" }
}
Write-Host "      $Model ready" -ForegroundColor Green

Write-Host "[4/5] GPU health" -ForegroundColor Cyan
$q = "clocks.sm,clocks.max.sm,power.draw,power.max_limit,temperature.gpu,memory.used,memory.total,clocks_throttle_reasons.active"
$g = (nvidia-smi --query-gpu=$q --format=csv,noheader,nounits) -split ',' | ForEach-Object { $_.Trim() }
$sm = [int]$g[0]; $smMax = [int]$g[1]; $temp = [int]$g[4]
$mask = [Convert]::ToInt64($g[7], 16)

$reasons = @{
    0x001 = "GPU idle"; 0x002 = "applications clocks setting"; 0x004 = "SW power cap"
    0x008 = "HW slowdown"; 0x010 = "sync boost"; 0x020 = "SW thermal slowdown"
    0x040 = "HW thermal slowdown"; 0x080 = "HW power brake"; 0x100 = "display clock setting"
}
$active = $reasons.Keys | Where-Object { $mask -band $_ } | ForEach-Object { $reasons[$_] }

Write-Host ("      clock   {0} / {1} MHz" -f $sm, $smMax)
Write-Host ("      power   {0} / {1} W" -f $g[2], $g[3])
Write-Host ("      temp    {0} C" -f $temp)
Write-Host ("      vram    {0} / {1} MiB" -f $g[5], $g[6])
Write-Host ("      throttle 0x{0:x} {1}" -f $mask, $(if ($active) { "(" + ($active -join ", ") + ")" } else { "(none)" }))

# Idle temperature is the honest signal for a cooling problem: under load anything runs hot, but a
# card sitting at 70 C with no work is not shedding heat at all.
if ($temp -ge 65) {
    Write-Host "      WARNING: $temp C is high. If the GPU is idle this indicates a cooling problem," -ForegroundColor Yellow
    Write-Host "               which will pin clocks at ~300 MHz and make a long run many times slower." -ForegroundColor Yellow
}
if ($mask -band 0x060) {
    Write-Host "      WARNING: thermal slowdown is ACTIVE - clean the cooling before a long run." -ForegroundColor Yellow
}

if ($Benchmark) {
    Write-Host "[5/5] measuring throughput (~1 min)" -ForegroundColor Cyan
    $body = @{
        model = $Model; stream = $false; think = $false
        messages = @(@{ role = "user"; content = "Write 300 words about knee MRI anatomy. Prose only." })
    } | ConvertTo-Json -Depth 5
    $r = Invoke-RestMethod http://localhost:11434/api/chat -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 1800
    $rate = [math]::Round($r.eval_count / ($r.eval_duration / 1e9), 1)

    $proc = (& $ollama ps | Select-String $Model) -replace '\s+', ' '
    Write-Host "      $rate tok/s   [$proc]"

    # 4,407 reports x ~200 output tokens in --no-evidence mode.
    $hours = [math]::Round((4407 * 200) / $rate / 3600, 1)
    Write-Host ("      projected full corpus (--no-evidence): {0} h" -f $hours) -ForegroundColor $(
        if ($hours -le 12) { "Green" } elseif ($hours -le 30) { "Yellow" } else { "Red" })
    if ($rate -lt 10) {
        Write-Host "      Below 10 tok/s the full corpus is not practical. Fix cooling, or use a" -ForegroundColor Yellow
        Write-Host "      smaller model (qwen3:8b / qwen3:4b)." -ForegroundColor Yellow
    }
} else {
    Write-Host "[5/5] skipping benchmark (pass -Benchmark to measure)" -ForegroundColor DarkGray
}

Write-Host "`nready.`n" -ForegroundColor Green

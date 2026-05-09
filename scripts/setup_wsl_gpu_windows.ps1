$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "== Windows GPU =="
$gpu = Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion, Status
$gpu | Format-Table -AutoSize

Write-Host "`n== NVIDIA-SMI =="
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
  throw "nvidia-smi.exe was not found on Windows PATH. Install or repair the Windows NVIDIA driver."
}
& $nvidiaSmi.Source

Write-Host "`n== WSL =="
wsl.exe --version
wsl.exe --status

Write-Host "`n== Update WSL =="
wsl.exe --update

Write-Host "`n== Restart WSL =="
Write-Host "The next command stops all WSL distros. Reopen Ubuntu after it finishes."
wsl.exe --shutdown

Write-Host "`nAfter reopening Ubuntu, run:"
Write-Host "  cd /home/amdev/sota/minimal-shot-av"
Write-Host "  .venv-v20/bin/python scripts/check_cuda_preflight.py --require-cuda"

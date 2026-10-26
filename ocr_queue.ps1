# Entry point of the right-click menu (started hidden by Explorer, see context_menu.ps1).
# Explorer starts ONE process per selected file. The first of them becomes the "leader": it collects the
# files that all processes put into a queue and OCRs them in a single run in a visible console, so the
# models are loaded once and there is one window and one summary for the whole selection.
param(
    [string]$Path,
    [string]$Opts = '',
    [string]$ListFile  # internal: the leader's visible run
)

$ErrorActionPreference = 'Stop'

if ($ListFile) {
    # ---- visible run
    $Host.UI.RawUI.WindowTitle = 'PDF OCR'
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONUTF8 = '1'
    $py = Join-Path $env:USERPROFILE '.pdf-ocr\venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $py)) {
        Write-Host 'The environment is not installed yet - run setup.bat first.'
        Read-Host 'Press Enter to close' | Out-Null
        exit 1
    }
    $paths = @(Get-Content -LiteralPath $ListFile -Encoding UTF8 | Where-Object { $_ })
    Remove-Item -LiteralPath $ListFile -Force -ErrorAction SilentlyContinue
    $optArgs = @($Opts -split '\s+' | Where-Object { $_ })
    Write-Host "OCR: $($paths.Count) item(s), options: $Opts"
    & $py (Join-Path $PSScriptRoot 'pdf_ocr.py') @paths @optArgs
    $rc = $LASTEXITCODE
    Write-Host ''
    if ($rc -ne 0) { Write-Host "Finished with errors (code $rc)." } else { Write-Host 'Finished.' }
    if (-not $env:PDF_OCR_NOPAUSE) { Read-Host 'Press Enter to close' | Out-Null }  # NOPAUSE: for tests
    exit $rc
}

# ---- queue mode (hidden)
$md5 = [System.Security.Cryptography.MD5]::Create()
$key = ([BitConverter]::ToString($md5.ComputeHash([Text.Encoding]::UTF8.GetBytes($Opts)))).Replace('-', '')
$qdir = Join-Path $env:TEMP "pdf-ocr-queue\$key"
New-Item -ItemType Directory -Force -Path $qdir | Out-Null
Set-Content -LiteralPath (Join-Path $qdir ([guid]::NewGuid().ToString() + '.txt')) -Value $Path -Encoding UTF8

$mutex = New-Object System.Threading.Mutex($false, "Local\pdf-ocr-$key")
Start-Sleep -Milliseconds 1500  # let Explorer start the other processes of the selection
$leader = $false
try { $leader = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $leader = $true }
if (-not $leader) { exit 0 }

try {
    while ($true) {
        Start-Sleep -Milliseconds 1500  # collect stragglers
        $files = @(Get-ChildItem -LiteralPath $qdir -Filter '*.txt' | Sort-Object CreationTime)
        if ($files.Count -eq 0) { break }
        $paths = @($files | ForEach-Object { Get-Content -LiteralPath $_.FullName -Encoding UTF8 } | Where-Object { $_ } | Select-Object -Unique)
        $files | Remove-Item -Force
        $list = Join-Path $qdir ('run-' + [guid]::NewGuid().ToString() + '.lst')
        Set-Content -LiteralPath $list -Value $paths -Encoding UTF8
        $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Opts', "`"$Opts`"", '-ListFile', "`"$list`"")
        Start-Process -FilePath 'powershell.exe' -ArgumentList $argList -Wait
    }
} finally {
    $mutex.ReleaseMutex()
}

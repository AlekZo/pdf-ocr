# Adds / removes a right-click "OCR" drop-down for PDF files and folders in Windows Explorer.
# Per-user (HKCU): no admin rights needed. Use install_context_menu.bat / uninstall_context_menu.bat.
param([ValidateSet('install', 'uninstall')][string]$Action = 'install')

$ErrorActionPreference = 'Stop'
$queue = Join-Path $PSScriptRoot 'ocr_queue.ps1'

# Edit this list to change the entries. Label, then options passed to pdf_ocr.py.
$presets = @(
    @{ Label = 'Russian (default)';              Args = '--lang ru' },
    @{ Label = 'Russian + tables and headings (slower)'; Args = '--lang ru --structure' },
    @{ Label = 'Russian, high quality (300 dpi)'; Args = '--lang ru --dpi 300' },
    @{ Label = 'Russian, fast (text only)';      Args = '--lang ru --fast --no-pdf' },
    @{ Label = 'Russian, redo existing';         Args = '--lang ru --overwrite' },
    @{ Label = 'Russian, fix rotated pages';     Args = '--lang ru --auto-rotate --textline-orient' },
    @{ Label = 'Russian, photo / curved pages';  Args = '--lang ru --unwarp --textline-orient' },
    @{ Label = 'English';                        Args = '--lang en' },
    @{ Label = 'Ukrainian';                      Args = '--lang uk' },
    @{ Label = 'Russian, force CPU';             Args = '--lang ru --device cpu' }
)

$roots = @(
    @{ Path = 'HKCU:\Software\Classes\SystemFileAssociations\.pdf\shell\PdfOcr'; Arg = '%1' },
    @{ Path = 'HKCU:\Software\Classes\Directory\shell\PdfOcr';                   Arg = '%1' }
)

foreach ($root in $roots) {
    if (Test-Path $root.Path) { Remove-Item $root.Path -Recurse -Force }
    if ($Action -eq 'uninstall') { continue }

    New-Item $root.Path -Force | Out-Null
    Set-ItemProperty $root.Path -Name 'MUIVerb' -Value 'OCR (PaddleOCR)'
    Set-ItemProperty $root.Path -Name 'Icon' -Value 'imageres.dll,-5302'
    # empty SubCommands = "this verb is a cascading menu built from the \shell subkeys"; without it Explorer
    # reports "This file does not have an app associated with it" on click
    Set-ItemProperty $root.Path -Name 'SubCommands' -Value ''
    New-Item "$($root.Path)\shell" -Force | Out-Null

    $i = 1
    foreach ($p in $presets) {
        $key = "$($root.Path)\shell\{0:D2}" -f $i
        New-Item "$key\command" -Force | Out-Null
        Set-ItemProperty $key -Name 'MUIVerb' -Value $p.Label
        # hidden launcher: the files of a multi-selection are collected and OCR'd in one run (ocr_queue.ps1)
        Set-ItemProperty "$key\command" -Name '(default)' -Value ('powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -Path "{1}" -Opts "{2}"' -f $queue, $root.Arg, $p.Args)
        $i++
    }
}

if ($Action -eq 'install') {
    Write-Host 'Installed. Right-click a PDF (Windows 11: "Show more options" or Shift+right-click) -> OCR (PaddleOCR).'
} else {
    Write-Host 'Removed the right-click OCR menu.'
}

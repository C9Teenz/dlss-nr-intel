<#
.SYNOPSIS
  Installs OptiScaler (XeSS on Intel Arc) into a DirectX 12 game folder,
  optionally together with the DLSS_NR_DX12.asi post-process plugin.

.DESCRIPTION
  - Copies the OptiScaler v0.9.4 release (third_party/optiscaler/pkg) into the game folder.
  - Renames OptiScaler.dll to a proxy name (default dxgi.dll).
  - Patches OptiScaler.ini with an Intel Arc preset:
      Dx12Upscaler=xess, Dxgi spoofing on (exposes the DLSS option in-game),
      optional XeSS Frame Generation (XeFG) + XeLL via fakenvapi.
  - Optionally copies plugins\DLSS_NR_DX12.asi + ort/ + dlss_nr_neural.ini + models/neural/ and enables ASI loading.
  - Writes uninstall_optiscaler_intel.ps1 into the game folder (restores backed-up files).

.EXAMPLE
  .\tools\install_optiscaler_intel.ps1 -GamePath "D:\Games\Cyberpunk 2077\bin\x64"
  .\tools\install_optiscaler_intel.ps1 -GamePath "...\Binaries\Win64" -FrameGen -WithNrFilter
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$GamePath,
    [ValidateSet('dxgi.dll', 'winmm.dll', 'version.dll', 'dbghelp.dll', 'd3d12.dll', 'wininet.dll', 'winhttp.dll')]
    [string]$ProxyName = 'dxgi.dll',
    [switch]$FrameGen,       # XeFG (Intel XeSS Frame Generation) using upscaler as FG input
    [switch]$WithNrFilter,   # also install plugins\DLSS_NR_DX12.asi (F11 toggles it in-game)
    [switch]$NoSpoofing      # use when the game already exposes XeSS/FSR natively
)

$ErrorActionPreference = 'Stop'
$repo   = Split-Path -Parent $PSScriptRoot
$pkg    = Join-Path $repo 'third_party\optiscaler\pkg'
$nrDist = Join-Path $repo 'dist\game-drop-in'

if (-not (Test-Path (Join-Path $pkg 'OptiScaler.dll'))) {
    throw "OptiScaler package not found at $pkg. Extract Optiscaler_0.9.4 .7z there first (tar -xf optiscaler.7z -C pkg)."
}
$GamePath = (Resolve-Path $GamePath).Path
if (-not (Get-ChildItem -Path $GamePath -Filter *.exe -File -ErrorAction SilentlyContinue)) {
    throw "No .exe in $GamePath. Point -GamePath at the folder containing the game executable (UE games: <Name>\Binaries\Win64)."
}
if (Test-Path (Join-Path $GamePath 'Engine')) {
    Write-Warning "Found an 'Engine' folder: for Unreal Engine games use <Codename>\Binaries\Win64 instead."
}

# ---------------------------------------------------------------- backup / manifest
$stamp     = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupDir = Join-Path $GamePath "_optiscaler_backup_$stamp"
$installed = New-Object System.Collections.Generic.List[string]

function Copy-WithBackup([string]$src, [string]$relDst) {
    $dst = Join-Path $GamePath $relDst
    if (Test-Path $dst) {
        $bk = Join-Path $backupDir $relDst
        New-Item -ItemType Directory -Force (Split-Path $bk) | Out-Null
        Copy-Item $dst $bk -Force
    }
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    Copy-Item $src $dst -Force
    $installed.Add($relDst)
}

# Our old standalone dxgi.dll proxy would collide with OptiScaler's proxy name.
$existingProxy = Join-Path $GamePath $ProxyName
if (Test-Path $existingProxy) {
    $orig = (Get-Item $existingProxy).VersionInfo.OriginalFilename
    Write-Host "Existing $ProxyName (OriginalFilename='$orig') will be backed up to $backupDir"
}

# ---------------------------------------------------------------- copy OptiScaler
Get-ChildItem -Path $pkg -Recurse -File | Where-Object {
    $_.Name -notin @('setup_windows.bat', 'setup_linux.sh') -and $_.Name -notlike '!!*'
} | ForEach-Object {
    $rel = $_.FullName.Substring($pkg.Length + 1)
    if ($rel -eq 'OptiScaler.dll') { $rel = $ProxyName }
    Copy-WithBackup $_.FullName $rel
}

# ---------------------------------------------------------------- optional NR filter plugin
if ($WithNrFilter) {
    # the DLSS-NR neural bridge (tools\package_neural_dropin.ps1 lays this out)
    Copy-WithBackup (Join-Path $nrDist 'plugins\DLSS_NR_DX12.asi') 'plugins\DLSS_NR_DX12.asi'
    Get-ChildItem (Join-Path $nrDist 'ort') -File | ForEach-Object { Copy-WithBackup $_.FullName "ort\$($_.Name)" }
    Copy-WithBackup (Join-Path $nrDist 'dlss_nr_neural.ini') 'dlss_nr_neural.ini'
    Get-ChildItem (Join-Path $nrDist 'models\neural') -File -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-WithBackup $_.FullName "models\neural\$($_.Name)" }
    if (-not (Get-ChildItem (Join-Path $nrDist 'models\neural') -File -ErrorAction SilentlyContinue)) {
        Write-Warning "no ONNX models in $nrDist\models\neural: the NR plugin will stay inactive (see PROGRESS-neural.md)"
    }
}

# ---------------------------------------------------------------- patch OptiScaler.ini (section aware)
$iniPath = Join-Path $GamePath 'OptiScaler.ini'
$lines   = [System.Collections.Generic.List[string]](Get-Content $iniPath)

function Set-Ini([string]$section, [string]$key, [string]$value) {
    $inSec = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $l = $lines[$i].Trim()
        if ($l -match '^\[(.+)\]$') { $inSec = ($Matches[1] -eq $section); continue }
        if ($inSec -and $l -match "^$([regex]::Escape($key))\s*=") {
            $lines[$i] = "$key=$value"; return
        }
    }
    throw "OptiScaler.ini: key [$section] $key not found"
}

Set-Ini 'Upscalers' 'Dx12Upscaler' 'xess'
Set-Ini 'Upscalers' 'Dx11Upscaler' 'xess'          # native DX11 XeSS (Arc only)
Set-Ini 'Spoofing'  'Dxgi'         ($(if ($NoSpoofing) { 'false' } else { 'true' }))
Set-Ini 'Spoofing'  'UEIntelAtomics' 'true'        # avoids "DX12 not supported" in UE games when spoofing on Intel
Set-Ini 'NvApi'     'OverrideNvapiDll' 'true'      # fakenvapi: Reflex -> XeLL
Set-Ini 'Menu'      'OverlayMenu'  'true'
Set-Ini 'Log'       'LogToFile'    'true'

if ($FrameGen) {
    Set-Ini 'FrameGen' 'Enabled'  'true'
    Set-Ini 'FrameGen' 'FGInput'  'upscaler'
    Set-Ini 'FrameGen' 'FGOutput' 'xefg'
}
if ($WithNrFilter) {
    Set-Ini 'Plugins' 'LoadAsiPlugins' 'true'
}
[System.IO.File]::WriteAllLines($iniPath, $lines, (New-Object System.Text.UTF8Encoding($false)))  # no BOM

# ---------------------------------------------------------------- uninstaller
$manifest  = ($installed | ForEach-Object { "    '$($_ -replace "'", "''")'" }) -join ",`r`n"
$uninstall = @"
# Auto-generated by dlss-nr-intel/tools/install_optiscaler_intel.ps1 on $stamp
`$ErrorActionPreference = 'Continue'
`$game = `$PSScriptRoot
`$files = @(
$manifest
)
foreach (`$f in `$files) { Remove-Item (Join-Path `$game `$f) -Force -ErrorAction SilentlyContinue }
`$bk = Join-Path `$game '_optiscaler_backup_$stamp'
if (Test-Path `$bk) {
    Get-ChildItem `$bk -Recurse -File | ForEach-Object {
        `$rel = `$_.FullName.Substring(`$bk.Length + 1)
        Copy-Item `$_.FullName (Join-Path `$game `$rel) -Force
    }
}
foreach (`$d in 'D3D12_Optiscaler','Licenses','plugins','ort','models\neural','models','compute') {
    `$p = Join-Path `$game `$d
    if ((Test-Path `$p) -and -not (Get-ChildItem `$p -Recurse -File)) { Remove-Item `$p -Recurse -Force }
}
Write-Host 'OptiScaler (Intel preset) removed.'
Remove-Item `$PSCommandPath -Force
"@
Set-Content -Path (Join-Path $GamePath 'uninstall_optiscaler_intel.ps1') -Value $uninstall -Encoding UTF8

Write-Host ""
Write-Host "OptiScaler v0.9.4 installed to: $GamePath" -ForegroundColor Green
Write-Host "  Proxy        : $ProxyName"
Write-Host "  Upscaler     : XeSS (DX12/DX11)"
Write-Host "  DXGI spoofing: $(-not $NoSpoofing)  (pick DLSS in the game menu; OptiScaler routes it to XeSS)"
Write-Host "  Frame Gen    : $(if ($FrameGen) { 'XeFG (input: upscaler)' } else { 'off' })"
Write-Host "  NR filter ASI: $(if ($WithNrFilter) { 'on (F11 toggles)' } else { 'off' })"
Write-Host "  Overlay menu : Insert key"
Write-Host "  Uninstall    : $GamePath\uninstall_optiscaler_intel.ps1"

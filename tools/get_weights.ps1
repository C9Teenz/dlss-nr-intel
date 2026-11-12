<#
.SYNOPSIS
  Extract the DLSS-NR logical weights from YOUR OWN nvngx_dlssnr.dll (N0).

.DESCRIPTION
  This script contains, downloads and redistributes no NVIDIA binary and no weights. It
  drives the third-party Apache-2.0 extractor MLX-DLSS (pinned below) against a DLL you
  supply and writes the result into the git-ignored work\ tree:

    work\mlxw\dlssnr-packed.safetensors    153 packed tensors (container as stored)
    work\mlxw\dlssnr-logical.safetensors   649 logical tensors (what the graph reads)

  NVIDIA's DLSS licence terms may forbid extraction. The result is for local use only:
  never commit it, never share it.

.EXAMPLE
  .\tools\get_weights.ps1 -Dll "D:\Games\SomeGame\nvngx_dlssnr.dll"
  .\tools\get_weights.ps1 -Dll ... -Golden      # also run the numpy oracle (tools\hnet\oracle.py)
  .\tools\get_weights.ps1 -Dll ... -Models      # and everything the runner needs (tools\hnet\make_models.py)
#>
param(
    [Parameter(Mandatory = $true)][string]$Dll,
    [string]$Python = "python",
    [switch]$Golden,
    [switch]$Models
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Work = Join-Path $Root "work"
$Mlx = Join-Path $Work "mlx-dlss"
$MlxUrl = "https://github.com/iamwavecut/MLX-DLSS.git"
$MlxPin = "06a3e11a8b68817127406ace5c764463543f699b"
$Tools = Join-Path $Mlx "python\mlxdlss\tools"
$Out = Join-Path $Work "mlxw"
$Packed = Join-Path $Out "dlssnr-packed.safetensors"
$Logical = Join-Path $Out "dlssnr-logical.safetensors"

if (-not (Test-Path -LiteralPath $Dll -PathType Leaf)) {
    throw "DLL not found: $Dll. Supply your own nvngx_dlssnr.dll (build 310.8.0.0 or newer); this project does not ship it."
}
$version = (Get-Item -LiteralPath $Dll).VersionInfo.FileVersion
Write-Host "nvngx_dlssnr.dll version: $version"
try {
    if ([version]($version -replace ',', '.' -replace '\s', '') -lt [version]"310.8.0.0") {
        Write-Warning "build older than 310.8.0.0; the recovered graph was validated on 310.8.0.0"
    }
} catch {
    Write-Warning "could not parse the DLL version '$version'"
}

if (-not (Test-Path (Join-Path $Tools "extract_dlssnr_weights.py"))) {
    if ((Test-Path $Mlx) -and (Get-ChildItem $Mlx -Force | Select-Object -First 1)) {
        throw "$Mlx exists but has no extractor; move it aside and re-run."
    }
    New-Item -ItemType Directory -Force $Work | Out-Null
    git clone $MlxUrl $Mlx
    if ($LASTEXITCODE -ne 0) { throw "git clone of MLX-DLSS failed" }
}
$head = (git -C $Mlx rev-parse HEAD).Trim()
if ($head -ne $MlxPin) {
    git -C $Mlx checkout $MlxPin
    if ($LASTEXITCODE -ne 0) { throw "could not pin MLX-DLSS at $MlxPin" }
}

& $Python -c "import numpy, safetensors" 2>$null
if ($LASTEXITCODE -ne 0) { throw "python needs numpy and safetensors: $Python -m pip install numpy safetensors" }

New-Item -ItemType Directory -Force $Out | Out-Null
& $Python (Join-Path $Tools "extract_dlssnr_weights.py") $Dll $Packed
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Packed)) { throw "weight extraction failed" }
& $Python (Join-Path $Tools "unpack_dlssnr_weights.py") $Packed $Logical
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Logical)) { throw "weight unpacking failed" }

# 649 tensors, fully_logical, 145 755 123 parameters
& $Python (Join-Path $Root "tools\hnet\common.py") --check-weights
if ($LASTEXITCODE -ne 0) { throw "the logical weights failed the check; re-extract from a clean DLL" }

if ($Models) {
    # the weight blob for the native runner, reference dumps, the golden, the ONNX fallbacks
    $env:PYTHONIOENCODING = "utf-8"
    & $Python (Join-Path $Root "tools\hnet\make_models.py") --weights real
    if ($LASTEXITCODE -ne 0) { throw "make_models failed" }
    Write-Host "Next: python tools\run_neural_checks.py --real"
} elseif ($Golden) {
    & $Python (Join-Path $Root "tools\hnet\oracle.py")
    if ($LASTEXITCODE -ne 0) { throw "oracle run failed" }
}
Write-Host "OK: $Logical"

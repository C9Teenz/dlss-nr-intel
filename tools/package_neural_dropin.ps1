<#
.SYNOPSIS
  Assemble dist\game-drop-in for the DLSS-NR neural bridge (N3).

.DESCRIPTION
  After building (cmake --build build), lays out:

    dist\game-drop-in\
      dxgi.dll                     standalone proxy (from the build)
      plugins\DLSS_NR_DX12.asi     the same module as an OptiScaler ASI plugin
      ort\onnxruntime.dll          ONNX Runtime 1.24.4, DirectML build (MIT)
      ort\DirectML.dll             DirectML 1.15.4 (Microsoft redistributable)
      dlss_nr_neural.ini           render_scale, intensity, profile, temporal, model_tag
      models\neural\*.onnx         ONLY with -IncludeModels: the ONNX files from work\onnx

  The ONNX files exported from real weights embed weights extracted from your own
  nvngx_dlssnr.dll. They are for your machine only: copy them into your own game folder,
  never into anything you share. Without -IncludeModels the folder is left empty.

.EXAMPLE
  .\tools\package_neural_dropin.ps1
  .\tools\package_neural_dropin.ps1 -IncludeModels -Tag nq-exp
#>
param(
    [switch]$IncludeModels,
    [string]$Tag = "nq-exp",
    [switch]$Synthetic
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dist = Join-Path $Root "dist\game-drop-in"
$OrtNative = Join-Path $Root "third_party\onnxruntime\extracted\runtimes\win-x64\native"
$DirectML = Join-Path $Root "third_party\directml\extracted\bin\x64-win\DirectML.dll"

$dxgi = Join-Path $Dist "dxgi.dll"
if (-not (Test-Path $dxgi)) { throw "build first: $dxgi missing (cmake --build build --target dxgi_proxy)" }
foreach ($d in 'plugins', 'ort', 'models\neural') { New-Item -ItemType Directory -Force (Join-Path $Dist $d) | Out-Null }

Copy-Item $dxgi (Join-Path $Dist "plugins\DLSS_NR_DX12.asi") -Force
foreach ($f in 'onnxruntime.dll', 'onnxruntime_providers_shared.dll') {
    Copy-Item (Join-Path $OrtNative $f) (Join-Path $Dist "ort\$f") -Force
}
Copy-Item $DirectML (Join-Path $Dist "ort\DirectML.dll") -Force

$ini = Join-Path $Dist "dlss_nr_neural.ini"
if (-not (Test-Path $ini)) {
    @"
[neural]
; this file is re-read while the game runs: save it and the change applies within half a second
; (logged; precision, model_tag and synthetic only after a restart)
; fraction of the back buffer the network sees; the field (and the ONNX model) follows from it
render_scale=0.5
; 0 = the game's frame, 1 = the model's picture, up to 2 extrapolates
intensity=1.0
; standard | natural | cinematic | neutral
profile=standard
; feed the previous prediction back (temporal gate). 0 for now: without motion vectors the
; history smears moving pictures (first game test, 2026-11-04)
temporal=0
; 2-4 = the network spread over that many frames: about 1/N of its GPU time per frame (1440p,
; render_scale 0.5: 36 / 19 / 13 / 10 ms for 1-4). Its last correction is moved along the
; picture's motion and faded where something new shows (warp=0: shown as it is, it lags)
interval=1
warp=1
; interval > 1: 1 = the moved correction follows this frame's edges through its local colour
; slope (less ghosting on cloth and limbs); 0 = the correction moved as it is
warp_model=1
; interval > 1 with warp_model=1: a new correction's weight against the last one carried onto its
; picture (1 = the new one alone). 0.5 halves the jump each new correction brings (less flicker)
warp_blend=0.5
; interval > 1: 1 = also search the motion coarsely over a wide range, so a fast camera that turns
; back, starts or stops keeps its correction (0 = only around the predicted motion)
warp_pyramid=1
; interval > 1: 1 = follow the motion frame to frame up to the frame the correction came from (a
; fast camera no longer moves it beyond the search: patches of correction from the wrong place)
warp_chain=0
; interval > 1 with warp_model=1: from this speed (field pixels a frame; at 1440p and render_scale
; 0.5 one is ~2 screen pixels) the moved correction's detail fades to a smooth regional one, gone
; at warp_fade_full: no ghost or swimming under a fast camera, the full effect back when it slows
; (warp_fade_full=0 = never; lower values fade sooner)
warp_fade_start=2
warp_fade_full=6
; what the fade goes to: regional (a smooth correction from this frame's colour) or none (the
; game's own picture: under a fast camera nothing from an older frame fits)
warp_fade_to=regional
; saved as a new number N > 0 while the game runs: the next N frames (the game's picture and the
; shown one, ~30 MB a frame at 1440p) go to dlss_nr_capture\<date-time> next to the log, for offline
; analysis (a beep when done); change the number again for another capture
capture=0
; fast = without the vendor's FP8 (E4M3) roundings: about 20 % less GPU time, less faithful.
; accurate = with them: closest to the real DLSS-NR (its correction jumps between frames; the
; stabiliser below calms it)
precision=fast
; stabiliser: the network's correction averaged over frames along the picture's motion (a new one
; weighs this much; 0 = off): against flicker and the shimmer of moving pictures. Where the moved
; picture does not match (a cut, something uncovered) the new correction is used alone. The
; frame's own detail is never averaged.
stabilize=0.05
; how much colour change (after following the motion) still counts as the same picture: raise
; (e.g. 2) if a still scene still shimmers (film grain, jitter), lower (0.5) if something drags
; behind moving things (2 made ghosting visible in the first game test)
stabilize_tolerance=1.0
; where anything moves, the least weight of the new correction: 0.5 = what was carried along
; halves every frame (no smear); lower (0.3) calms more but may smear on turning faces, 1 = no
; averaging while moving
stabilize_motion=0.5
; numerics of the export: nq-exp (fast) or vk-pwl (closest to the vendor)
model_tag=$Tag
; 1 = use *_synthetic.onnx (plumbing test only, the picture is meaningless)
synthetic=$(if ($Synthetic) { 1 } else { 0 })
"@ | Set-Content -Path $ini -Encoding ASCII
}

# legacy files of the retired 5-stage placeholder graph
foreach ($old in 'models\dlss_nr_xe2_weights.bin', 'compute\inference_graph.hlsl', 'compute\matmul.hlsl',
                 'compute\image_process.hlsl', 'compute\vector_add.hlsl') {
    $p = Join-Path $Dist $old
    if (Test-Path $p) { Remove-Item $p -Force }
}

if ($IncludeModels) {
    # the native graph needs only the weight blob (hnet-ends_nq-exp.bin, tools\hnet\export_hybrid.py);
    # whole-graph ONNX files are the fallback for other tags and need ONNX Runtime
    $suffix = if ($Synthetic) { "_synthetic" } else { "" }
    $models = @(Get-ChildItem (Join-Path $Root "work\onnx") -File -ErrorAction SilentlyContinue | Where-Object {
        ($_.Name -eq "hnet-ends_nq-exp$suffix.bin") -or ($_.Name -like "hnet_*_${Tag}_fp16$suffix.onnx")
    } | Where-Object { $Synthetic -or $_.Name -notlike '*_synthetic*' })
    if (-not $models) { throw "no models in work\onnx; run tools\hnet\export_hybrid.py (and/or export_onnx.py)" }
    if (-not $Synthetic) {
        Write-Warning "These ONNX files contain weights extracted from your nvngx_dlssnr.dll. Local use only - never share them."
    }
    $models | ForEach-Object { Copy-Item $_.FullName (Join-Path $Dist "models\neural\$($_.Name)") -Force; Write-Host "  model $($_.Name)" }
}

Write-Host "Packaged $Dist"
Write-Host "  copy dxgi.dll (or plugins\DLSS_NR_DX12.asi with OptiScaler), ort\, dlss_nr_neural.ini and models\neural\ next to the game .exe"

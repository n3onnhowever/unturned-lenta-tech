param(
    [Parameter(Mandatory = $true)]
    [string]$Video,

    [string]$OutCsv = "output\submission.csv",
    [string]$WorkDir = "work",
    [string]$Weights = "weights\price_tag_holdout_43_15_neg_e12_best.pt",
    [string]$UpscaleModel = "",
    [string]$UpscaleModelName = "edsr",
    [int]$UpscaleScale = 4,
    [double]$Conf = 0.01,
    [int]$ImgSize = 1280
)

$ErrorActionPreference = "Stop"

$videoPath = Resolve-Path $Video
$workRoot = New-Item -ItemType Directory -Force $WorkDir
$framesRoot = Join-Path $workRoot.FullName "frames"
$detectDir = Join-Path $workRoot.FullName "detect"
$pipelineDir = Join-Path $workRoot.FullName "pipeline"

python scripts\extract_video_frames.py -i $videoPath -o $framesRoot --stride 1

$frameDirs = Get-ChildItem -Directory $framesRoot | Sort-Object LastWriteTime -Descending
if (-not $frameDirs) {
    throw "No frame directory was created under $framesRoot"
}
$framesDir = $frameDirs[0].FullName

python scripts\detect_price_tags_trained.py `
    --weights $Weights `
    --source $framesDir `
    --output $detectDir `
    --conf $Conf `
    --imgsz $ImgSize

$detectionsCsv = Join-Path $detectDir "detections.csv"

$upscaleArgs = @()
if ($UpscaleModel -ne "") {
    $upscaleArgs = @(
        "--upscale-model", (Resolve-Path $UpscaleModel),
        "--upscale-model-name", $UpscaleModelName,
        "--upscale-scale", $UpscaleScale
    )
}

python scripts\lenta_hackathon_pipeline.py `
    --video $videoPath `
    --frames $framesDir `
    --detections $detectionsCsv `
    --out-csv $OutCsv `
    --work-dir $pipelineDir `
    --engine paddle `
    @upscaleArgs

Write-Output "CSV written to $OutCsv"

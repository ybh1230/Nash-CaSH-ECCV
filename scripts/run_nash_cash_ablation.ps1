param(
    [int]$TargetHeight = 1088,
    [int]$TargetWidth = 1920,
    [int]$CacheSteps = 2,
    [string]$PromptFile = "",
    [string]$OutputDir = "outputs\nash_cash_ablation"
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$prompt = "A realistic close-up of an elderly man with gray hair and a thick gray beard, wearing a light-colored shirt. His head is slightly lowered. The camera zooms from full body to close-up, highlighting detailed facial wrinkles, skin texture, forehead lines, eye bags, and beard strands. High resolution, cinematic lighting, sharp details."
if ($PromptFile -ne "" -and (Test-Path -LiteralPath $PromptFile)) {
    $prompt = Get-Content -LiteralPath $PromptFile -Raw
}

$common = @(
    "--target_height", $TargetHeight,
    "--target_width", $TargetWidth,
    "--prompt", $prompt
)

python inference.py @common --mode nocache --no-nash_cash --output "$OutputDir\baseline_override.mp4"
python inference.py @common --mode nocache --nash_cash --output "$OutputDir\nash_cash_nocache.mp4"
python inference.py @common --mode cache --cache_steps $CacheSteps --nash_cash --output "$OutputDir\nash_cash_cache_p$CacheSteps.mp4"
python inference.py @common --mode cache --cache_steps 5 --nash_cash --output "$OutputDir\nash_cash_cache_p5.mp4"

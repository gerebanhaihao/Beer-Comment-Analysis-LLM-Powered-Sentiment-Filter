param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $project
$dist = Join-Path $project "dist"
$staging = Join-Path $dist "_build"

& $Python -m PyInstaller --noconfirm --onedir --windowed `
    --name BeerSentiment `
    --distpath $staging `
    --paths (Join-Path $project "src") `
    --exclude-module sentence_transformers `
    --exclude-module torch `
    --exclude-module transformers `
    --exclude-module sklearn `
    --exclude-module scipy `
    --exclude-module pandas `
    --exclude-module matplotlib `
    --exclude-module IPython `
    --exclude-module pytest `
    (Join-Path $project "src\beer_sentiment\gui.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败" }

$newRelease = Join-Path $staging "BeerSentiment"
Copy-Item (Join-Path $project "config") (Join-Path $newRelease "config") -Recurse -Force
Copy-Item (Join-Path $project "prompts") (Join-Path $newRelease "prompts") -Recurse -Force
Copy-Item (Join-Path $project "benchmark") (Join-Path $newRelease "benchmark") -Recurse -Force
New-Item -ItemType Directory -Force -Path (Join-Path $newRelease "incoming"), (Join-Path $newRelease "data"), (Join-Path $newRelease "output") | Out-Null

$release = Join-Path $dist "BeerSentiment"
if (Test-Path -LiteralPath $release) {
    $backup = Join-Path $dist ("BeerSentiment-backup-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    Move-Item -LiteralPath $release -Destination $backup
    Write-Host "旧版本已保留：$backup"
}
Move-Item -LiteralPath $newRelease -Destination $release
$intermediateExe = Join-Path $project "build\BeerSentiment\BeerSentiment.exe"
if (Test-Path -LiteralPath $intermediateExe) {
    Rename-Item -LiteralPath $intermediateExe -NewName "BeerSentiment.build-only"
}
Write-Host "完成：$release\BeerSentiment.exe"

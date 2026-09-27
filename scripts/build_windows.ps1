param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $project
$dist = Join-Path $project "dist"
$staging = Join-Path $dist "_build"
$work = Join-Path $staging "_work"

& $Python -m PyInstaller --noconfirm --onedir --windowed `
    --name BeerSentiment `
    --distpath $staging `
    --workpath $work `
    --specpath $staging `
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
$realBenchmark = Join-Path $project "benchmark\beer_sentiment_benchmark_real.jsonl"
if (-not (Test-Path -LiteralPath $realBenchmark -PathType Leaf)) {
    throw "缺少本地真实评测集：$realBenchmark"
}
Copy-Item (Join-Path $project "config") (Join-Path $newRelease "config") -Recurse -Force
Copy-Item (Join-Path $project "prompts") (Join-Path $newRelease "prompts") -Recurse -Force
New-Item -ItemType Directory -Force -Path (Join-Path $newRelease "benchmark"), (Join-Path $newRelease "data"), (Join-Path $newRelease "output"), (Join-Path $newRelease "artifacts") | Out-Null
Copy-Item -LiteralPath (Join-Path $project "benchmark\beer_sentiment_benchmark.jsonl") -Destination (Join-Path $newRelease "benchmark") -Force
Copy-Item -LiteralPath $realBenchmark -Destination (Join-Path $newRelease "benchmark") -Force

$release = Join-Path $dist "BeerSentiment"
if (Test-Path -LiteralPath $release) {
    $backup = Join-Path $dist ("BeerSentiment-backup-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Move-Item -LiteralPath $release -Destination $backup -ErrorAction Stop
            break
        } catch {
            if ($attempt -eq 5) { throw }
            Start-Sleep -Seconds 2
        }
    }
    Write-Host "旧版本已保留：$backup"
    foreach ($runtimeName in @("data", "output", "artifacts")) {
        $runtimeSource = Join-Path $backup $runtimeName
        if (Test-Path -LiteralPath $runtimeSource) {
            Copy-Item (Join-Path $runtimeSource "*") (Join-Path $newRelease $runtimeName) -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
Copy-Item -LiteralPath $newRelease -Destination $release -Recurse
Write-Host "完成：$release\BeerSentiment.exe"

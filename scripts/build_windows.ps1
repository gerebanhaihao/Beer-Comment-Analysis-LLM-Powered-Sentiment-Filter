param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $project
$dist = Join-Path $project "dist"
$staging = Join-Path $dist "_build"
$work = Join-Path $staging "_work"

& $Python -c "import sentence_transformers, torch, PyInstaller"
if ($LASTEXITCODE -ne 0) { throw "打包需要 sentence-transformers、torch 和 PyInstaller；请先安装项目的 [llm,rag] 依赖及 pyinstaller" }

$env:USE_TF = "0"
& $Python -m PyInstaller --noconfirm --onedir --windowed `
    --name BeerSentiment `
    --distpath $staging `
    --workpath $work `
    --specpath $staging `
    --paths (Join-Path $project "src") `
    --hidden-import sentence_transformers.models.Transformer `
    --hidden-import sentence_transformers.models.Pooling `
    --hidden-import sentence_transformers.models.Normalize `
    --copy-metadata sentence-transformers `
    --copy-metadata transformers `
    --exclude-module tensorflow `
    --exclude-module keras `
    --exclude-module tf_keras `
    --exclude-module jax `
    --exclude-module flax `
    --exclude-module torchvision `
    --exclude-module torchaudio `
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
$modelCache = Join-Path ([Environment]::GetFolderPath('UserProfile')) ".cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots"
$modelSnapshot = Get-ChildItem -LiteralPath $modelCache -Directory -ErrorAction Stop | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $modelSnapshot -or -not (Test-Path -LiteralPath (Join-Path $modelSnapshot.FullName "model.safetensors"))) {
    throw "缺少本地 bge-small-zh-v1.5 模型缓存：$modelCache"
}
$modelDest = Join-Path $newRelease "models\bge-small-zh-v1.5"
New-Item -ItemType Directory -Force -Path $modelDest | Out-Null
Copy-Item (Join-Path $modelSnapshot.FullName "*") $modelDest -Recurse -Force
$ragConfig = Join-Path $newRelease "config\rag.yaml"
$ragText = Get-Content -LiteralPath $ragConfig -Raw -Encoding UTF8
$ragText = $ragText.Replace("model: BAAI/bge-small-zh-v1.5", "model: models/bge-small-zh-v1.5")
Set-Content -LiteralPath $ragConfig -Value $ragText -Encoding UTF8
New-Item -ItemType Directory -Force -Path (Join-Path $newRelease "benchmark"), (Join-Path $newRelease "data"), (Join-Path $newRelease "output"), (Join-Path $newRelease "artifacts") | Out-Null
Copy-Item -LiteralPath (Join-Path $project "benchmark\beer_sentiment_benchmark.jsonl") -Destination (Join-Path $newRelease "benchmark") -Force
Copy-Item -LiteralPath $realBenchmark -Destination (Join-Path $newRelease "benchmark") -Force

$release = Join-Path $dist "BeerSentiment"
if (Test-Path -LiteralPath $release) {
    $backup = Join-Path $dist ("BeerSentiment-backup-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    $runtimeRoot = $release
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Move-Item -LiteralPath $release -Destination $backup -ErrorAction Stop
            $runtimeRoot = $backup
            Write-Host "旧版本已保留：$backup"
            break
        } catch {
            if ($attempt -eq 5) {
                $release = Join-Path $dist ("BeerSentiment-RAG-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
                Write-Warning "旧发布目录无法改名，新版本将保存到：$release"
            } else {
                Start-Sleep -Seconds 2
            }
        }
    }
    foreach ($runtimeName in @("data", "output", "artifacts")) {
        $runtimeSource = Join-Path $runtimeRoot $runtimeName
        if (Test-Path -LiteralPath $runtimeSource) {
            Copy-Item (Join-Path $runtimeSource "*") (Join-Path $newRelease $runtimeName) -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
Copy-Item -LiteralPath $newRelease -Destination $release -Recurse
Write-Host "完成：$release\BeerSentiment.exe"

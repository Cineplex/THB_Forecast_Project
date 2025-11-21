# Set the location for centralized pycache
$env:PYTHONPYCACHEPREFIX = "$PSScriptRoot\_pycache_store"

# Create the directory if it doesn't exist
if (-not (Test-Path $env:PYTHONPYCACHEPREFIX)) {
    New-Item -ItemType Directory -Path $env:PYTHONPYCACHEPREFIX | Out-Null
}

# Add project root to PYTHONPATH
$env:PYTHONPATH = $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

# Run the pipeline
Write-Host "🚀 Running pipeline with centralized pycache at: $env:PYTHONPYCACHEPREFIX"
python pipelines/run_pipeline.py

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Done."
} else {
    Write-Host "❌ Failed with exit code $LASTEXITCODE"
}

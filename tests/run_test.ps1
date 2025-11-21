# Helper script to run test scripts with proper environment setup
# Usage: .\tests\run_test.ps1 test_single_feature.py --feature gold --start 2024-08-01

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ScriptName,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Set up environment
$projectRoot = $PSScriptRoot | Split-Path
$env:PYTHONPATH = $projectRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPYCACHEPREFIX = Join-Path $projectRoot "_pycache_store"

# Build the full script path
$scriptPath = Join-Path $PSScriptRoot $ScriptName

if (-not (Test-Path $scriptPath)) {
    Write-Host "❌ Script not found: $scriptPath" -ForegroundColor Red
    exit 1
}

# Run the test script with arguments
Write-Host "🧪 Running test: $ScriptName" -ForegroundColor Cyan
python $scriptPath @Arguments

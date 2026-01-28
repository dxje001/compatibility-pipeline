# Compatibility Pipeline - Quick Start Script
# Run from PowerShell: .\start.ps1

Write-Host "=== Compatibility Pipeline Setup ===" -ForegroundColor Cyan

# Navigate to project directory
Set-Location $PSScriptRoot

# Check if venv exists, create if not
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -q numpy pandas scikit-learn scipy pyyaml joblib streamlit

# Check if models exist
if (-not (Test-Path "artifacts\runs\seed_11\models")) {
    Write-Host "Training models (first run only)..." -ForegroundColor Yellow
    python -m pipeline.run --config configs/config.yaml --seed 11
}

# Start the UI
Write-Host ""
Write-Host "=== Starting Streamlit UI ===" -ForegroundColor Green
Write-Host "Open http://localhost:8501 in your browser" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

streamlit run ui/app.py

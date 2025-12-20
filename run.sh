#!/bin/bash

# Voicet Runner Script
# This script prepares the environment and starts the Flask application.

set -e

echo "🚀 Preparing to run Voicet..."

# 1. Define paths
PROJECT_ROOT="$(pwd)"
VENV_BIN="$PROJECT_ROOT/venv/bin"
UPLOAD_DIR="$PROJECT_ROOT/Voicet/project/static/uploads"

# 2. Check for virtual environment
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 3. Create required directories
echo "📁 Ensuring upload directory exists..."
mkdir -p "$UPLOAD_DIR"

# 4. Check for system dependencies
echo "🔍 Checking system dependencies..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️ Warning: 'ffmpeg' is not installed. Video processing will fail."
fi
if ! command -v sox &> /dev/null; then
    echo "⚠️ Warning: 'sox' is not installed. Audio joining will fail."
fi

# 5. Check for Vakyansh TTS models (Hindi)
MODEL_CHECK_PATH="$PROJECT_ROOT/VAKYANSH_TTS/tts_infer/translit_models/hindi/female/glow_ckp"
if [ ! -d "$MODEL_CHECK_PATH" ]; then
    echo "ℹ️ Note: Hindi TTS models not found. You can download them using ./setup_models.sh"
fi

# 6. Run the application
echo "🌐 Starting Flask application..."
cd Voicet
export FLASK_APP=project
"$VENV_BIN/flask" run

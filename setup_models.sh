#!/bin/bash

# setup_models.sh - Automated model setup for Voicet

set -e

echo "🚀 Starting Voicet Model Setup..."

# 1. Install megacmd if not present
if ! command -v mega-get &> /dev/null; then
    echo "📦 Installing megacmd via Homebrew..."
    brew install megacmd
else
    echo "✅ megacmd is already installed."
fi

# 2. Create target directory
mkdir -p VAKYANSH_TTS/tts_infer/translit_models

# 3. Download models
echo "📥 Downloading models from Mega.nz (this may take a few minutes)..."
mega-get https://mega.nz/folder/VQlnHTiZ#WCUFo_ukvJbuMEWlfsUDPA VAKYANSH_TTS/tts_infer/translit_models/

echo "✅ Setup complete! You can now translate videos into Hindi."

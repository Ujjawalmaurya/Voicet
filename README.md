# Voicet
Step 1: 🎥 Voicet is an innovative Python-based application that facilitates the translation of videos from one language to another via a Progressive Web Application.

Step 2: 🚪 Upon accessing voicet.tech, users are directed to the Login Page. In the event that they don't possess an existing account, they can register via email by navigating to the Sign Up page located within the Navbar.

Step 3: 📝 To create an account, users are prompted to input their email address, username, and password. Upon completing registration, they will be directed to the Login Page, where they can input their login credentials to access their account.

Step 4: 📷 The Gallery exhibits all the videos that the User has posted. To upload a video, users must click on the Upload button located in the Navbar. Users have the option to download any YouTube short or upload any video of their own. After selecting their video, users must click on the Upload button.

Step 5: 🌐 To translate a video, users must select the Translate Button located on the desired video. They must then choose the language into which they wish to translate the video, as well as the gender of the audio. Once these preferences are selected, users must click on the Translate Button.

Step 6: 🤖 In order to translate the video, we generate captions from the video using OpenAI's Whisper STT (Speech To Text) Machine Learning Model. Next, we translate the English subtitles to the target language using Facebook's NLLB ML Model. Finally, we generate audio files from the translations utilizing Vakyansh TTS, merge the audio files, and superimpose them onto the original video.

Check it out at https://voicet.tech 🌎🗣️

# Setup & Installation on Linux

### 1. Prerequisites
Ensure you have Python 3.10+ and the following Linux tools installed:
```bash
sudo apt update && sudo apt install ffmpeg sox
```

### 2. Fetch VakyanshTTS Models
The models are required for the Hindi voice. You can automate the setup or do it manually.

**Option A: Automated Setup (Recommended)**
Install `megacmd` and run the download command:
```bash
# Install megacmd via Homebrew
brew install megacmd

# Download Hindi female models
# This will download the models directly to the required directory structure
mega-get https://mega.nz/folder/VQlnHTiZ#WCUFo_ukvJbuMEWlfsUDPA VAKYANSH_TTS/tts_infer/translit_models/
```

**Option B: Manual Setup**
1. Download the models from [Mega.nz](https://mega.nz/folder/VQlnHTiZ#WCUFo_ukvJbuMEWlfsUDPA).
2. Ensure the folder structure is:
   `VAKYANSH_TTS/tts_infer/translit_models/hindi/female/glow_ckp`
   `VAKYANSH_TTS/tts_infer/translit_models/hindi/female/hifi_ckp`


### 3. Environment Setup
Create a virtual environment and install the required Python packages:
```bash
# From the root of the repository
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Project Configuration
Ensure the upload directory exists:
```bash
mkdir -p Voicet/project/static/uploads
```

### 5. Running the Application
```bash
cd Voicet
export FLASK_APP=project
flask run
```
Then open your browser and navigate to `http://127.0.0.1:5000`.

# Screenshot
![Voicet Homepage](Voicet-Homepage.png)


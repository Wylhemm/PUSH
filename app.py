import os
from flask import Flask, request, jsonify, send_file
from TTS.api import TTS
import librosa
import soundfile as sf
import io

app = Flask(__name__)

# Hardcoded variables for model paths and configurations
MODEL_PATH = 'path/to/xttsv2.pth'
CONFIG_PATH = 'path/to/xttsv2_config.json'
VOCODER_PATH = 'path/to/vocoder.pth'
VOCODER_CONFIG_PATH = 'path/to/vocoder_config.json'

# Initialize TTS with the XTTSV2 model
tts = TTS(model_path=MODEL_PATH, 
          config_path=CONFIG_PATH, 
          vocoder_path=VOCODER_PATH, 
          vocoder_config_path=VOCODER_CONFIG_PATH)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data['text']

    # Read the MP3 file and convert it to WAV format
    speaker_mp3, _ = librosa.load('speaker.mp3', sr=22050)
    speaker_wav_bytes = io.BytesIO()
    sf.write(speaker_wav_bytes, speaker_mp3, 22050, format='wav')
    speaker_wav_bytes.seek(0)

    # Synthesize speech using the XTTSV2 model
    wav = tts.tts(text, speaker_wav=speaker_wav_bytes)

    # Save the generated audio to a file
    output_path = 'output.wav'
    tts.save_wav(wav, output_path)

    # Return the generated audio file
    return send_file(output_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
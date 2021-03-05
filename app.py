from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import noisereduce as nr
import base64

from io import BytesIO

# Spectrogram Imports
import librosa
import librosa.display
import numpy as np

import matplotlib.pyplot as plt
import soundfile as sf

from scipy.io import wavfile

@app.route('/api/uploadRecording', methods=['POST'])
def upload_recording():
    # Uploaded audio file is retreived 
    file = request.files['file']
    rate, data = wavfile.read(file)

    ## Librosa variables 
    n_fft= 2048 # Length of FFT Window
    hop_length = 100 # Number of samples between successive frames
    
    # Spectrogram of input voice recording #
    y, sr = librosa.load(file)

    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    librosa.display.specshow(
        mel_spect, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig('spectrogram.jpeg')

    image = BytesIO()
    plt.savefig(image, format='jpg')
    spectrogram_image = base64.encodestring(image.getvalue()).decode('utf-8')
    # #

    # Noise Reduction of input voice recording #

    data = data.flatten() / 32768  # Converting 16bit integer sound file to a 32 bit floating point
    # .flatten() fixed error "Invalid shape for monophonic audio" 
    noisy_part = data[:len(data)]          

    reduced_noise = nr.reduce_noise(
        audio_clip=data, noise_clip=noisy_part, verbose=False)

    wavfile.write("noise_reduced.wav", rate,
                  (reduced_noise * 32768).astype(np.int16))
    # Convert back to 16bit integer and save audio file
    ##

    # Spectrogram of noise reduced voice recording #
    fname_nr = "noise_reduced.wav"
    f_nr = open(fname_nr, "rb")
    nr_audio = base64.b64encode(f_nr.read()).decode('utf-8')

    y_nr, sr_nr = librosa.load('noise_reduced.wav')
    mel_spect_nr = librosa.feature.melspectrogram(
        y=y_nr, sr=sr_nr, n_fft=n_fft, hop_length=hop_length)
    mel_spect_nr = librosa.power_to_db(mel_spect_nr, ref=np.max)

    librosa.display.specshow(
        mel_spect_nr, sr=sr_nr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Noise Reduced Spectrogram')
    plt.savefig('noise_reduced.jpeg')

    image_nr = BytesIO()
    plt.savefig(image_nr, format='jpg')
    nr_spectrogram_image = base64.encodestring(image_nr.getvalue()).decode('utf-8')

    return jsonify({
        "spectrogram_image": spectrogram_image,
        "nr_spectrogram_image": nr_spectrogram_image,
        "nr_audio": nr_audio
    })
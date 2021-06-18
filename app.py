from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import noisereduce as nr
import base64
from io import BytesIO
import librosa
import librosa.display
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile
import pandas as pd
from sklearn.preprocessing import LabelEncoder

plt.switch_backend('agg')

@app.route('/api/uploadRecording', methods=['POST'])
def upload_recording():

    # Loading Model
    model = load_model('model_1.h5')

    # Labeles for Emotions
    lb = LabelEncoder()
    lb.fit(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

    # Uploaded audio file is retreived 
    file = request.files['file']
    rate, data = wavfile.read(file)

    
    # Spectrogram of input voice recording #
    y, sr = librosa.load(file, res_type='kaiser_best' ,duration=3,sr=44100,offset=0.5)

    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128,fmax=8000)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)


    log_spectrogram = np.mean(mel_spect, axis = 0)


    librosa.display.specshow(
        mel_spect, sr=sr, x_axis='time', y_axis='mel')

    plt.title('Spectrogram')
    # plt.colorbar(format='%+2.0f dB')
    plt.savefig('spectrogram.jpeg')

    image = BytesIO()
    plt.savefig(image, format='jpg')
    spectrogram_image = base64.encodestring(image.getvalue()).decode('utf-8')
    # #

    # # Get Prediction for Emotion
    # df = pd.DataFrame(columns=['mel_spectrogram'])
    # df.loc[0] = [log_spectrogram]
    # df = pd.DataFrame(df['mel_spectrogram'].values.tolist())

    # predicting_data= np.array(df.iloc[:, :])

    # mean = np.mean(predicting_data, axis=1)
    # std = np.std(predicting_data, axis=1)
    # predicting_data = (predicting_data - mean)/std

    # predictions = model.predict(predicting_data[:,:,np.newaxis])
    # predictions=    predictions.argmax(axis=1)
    # predictions = predictions.astype(int).flatten()
    # predictions = (lb.inverse_transform((predictions)))

    # print(predictions)
    # #


    empty_df = pd.DataFrame(index=np.arange(1), columns=np.arange(259))
    df = pd.DataFrame(columns=['mel_spectrogram'])

    df.loc[0] = [log_spectrogram]
    df_combined = pd.concat([pd.DataFrame(df['mel_spectrogram'].values.tolist()), empty_df], ignore_index=True)
    df_combined = df_combined.fillna(0)

    t_test= df_combined
    t_test_array = np.array(t_test)
    threed_test_array = t_test_array[:,:,np.newaxis]
    threed_test_array.shape



    predictions = model.predict(threed_test_array[:1])
    predictions = predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (lb.inverse_transform((predictions)))

    print(predictions)


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
        y=y_nr, sr=sr_nr)
    mel_spect_nr = librosa.power_to_db(mel_spect_nr, ref=np.max)

    librosa.display.specshow(
        mel_spect_nr, sr=sr_nr, x_axis='time', y_axis='mel')
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
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import noisereduce as nr
import base64
from io import BytesIO
import librosa
import librosa.display
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
from json import JSONEncoder

plt.switch_backend('agg')

class Prediction:
    def __init__(self, angry, disgust, fear, happy, neutral, sad, surprise):
        self.angry = angry.item()
        self.disgust= disgust.item()
        self.fear = fear.item()
        self.happy = happy.item()
        self.neutral = neutral.item()
        self.sad = sad.item()
        self.surprise = surprise.item()

# subclass JSONEncoder
class JSONEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__

@app.route('/api/uploadRecording', methods=['POST'])
def upload_recording():

    # Uploaded audio file is retreived 
    file = request.files['file']
    rate, data = wavfile.read(file)

    noise_reduce_check = request.form['noiseReduction']
    
    # Spectrogram of input voice recording #
    spectrogram_name = "Mel-Spectrogram"
    log_spectrogram1, spectrogram_image1 = get_mel_spectrogram(file, spectrogram_name)

    empty_df = pd.DataFrame(index=np.arange(1), columns=np.arange(259)) #259, 454
    df = pd.DataFrame(columns=['mel_spectrogram'])

    df.loc[0] = [log_spectrogram1]
    df_combined = pd.concat([pd.DataFrame(df['mel_spectrogram'].values.tolist()), empty_df], ignore_index=True)
    df_combined = df_combined.fillna(0)

    if(noise_reduce_check == 'false'):
        prediction = get_prediction(df_combined)
        return jsonify({
            "spectrogram_image": spectrogram_image1,
            "prediction" : JSONEncoder().encode(prediction)
        })

    ##########################################################################

    ##########################################################################

    elif(noise_reduce_check == 'true' or noise_reduce_check == 'true-recorded'):
        # Noise Reduction of input voice recording #
        data = data.flatten() / 32768   # Converting 16bit integer sound file to a 32 bit floating point
                                        # .flatten() fixed error "Invalid shape for monophonic audio" 
        noisy_part = data[:len(data)]          

        reduced_noise = nr.reduce_noise(
            audio_clip=data, noise_clip=noisy_part, verbose=False)

        wavfile.write("noise_reduced.wav", rate*2, (reduced_noise * 32768).astype(np.int16)) # Convert back to 16bit integer and save audio file

        # if(noise_reduce_check == 'true'):
        #     wavfile.write("noise_reduced.wav", rate*2, (reduced_noise * 32768).astype(np.int16)) # Convert back to 16bit integer and save audio file
        # if(noise_reduce_check == 'true-recorded'):
        #     wavfile.write("noise_reduced.wav", rate*2, (reduced_noise * 32768).astype(np.int16)) # Convert back to 16bit integer and save audio file

        # Spectrogram of noise reduced voice recording #
        fname_nr = "noise_reduced.wav"
        f_nr = open(fname_nr, "rb")
        audio_nr = base64.b64encode(f_nr.read()).decode('utf-8')


        spectrogram_name = "NR_Mel-Spectrogram"
        log_spectrogram_nr, spectrogram_image_nr = get_mel_spectrogram(fname_nr, spectrogram_name)

        empty_df = pd.DataFrame(index=np.arange(1), columns=np.arange(259))
        df = pd.DataFrame(columns=['mel_spectrogram'])

        df.loc[0] = [log_spectrogram_nr]
        df_combined = pd.concat([pd.DataFrame(df['mel_spectrogram'].values.tolist()), empty_df], ignore_index=True)
        df_combined = df_combined.fillna(0)

        prediction = get_prediction(df_combined)

        return jsonify({
            "spectrogram_image": spectrogram_image1,
            "prediction" : JSONEncoder().encode(prediction),
            "spectrogram_image_nr": spectrogram_image_nr,
            "audio_nr": audio_nr
        })


def get_mel_spectrogram(file, spectrogram_name):
    y, sr = librosa.load(file, res_type='kaiser_best' ,duration=5, sr=44100,offset=0.5 )
    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128,fmax=8000)

    librosa.display.specshow(
        librosa.power_to_db(mel_spect, ref=np.max), sr=sr, x_axis='time', y_axis='mel')


    log_spectrogram = np.mean(librosa.power_to_db(mel_spect), axis = 0)

    if(spectrogram_name == "Mel-Spectrogram"):
        plt.title('Mel-Spectrogram')
        plt.savefig('spectrogram.jpeg')

    elif(spectrogram_name == "NR_Mel-Spectrogram"):
        plt.title('Noise Reduced Mel-Spectrogram')
        plt.savefig('noise_reduced.jpeg')
        
    image_nr = BytesIO()
    plt.savefig(image_nr, format='jpg')
    spectrogram_image = base64.encodestring(image_nr.getvalue()).decode('utf-8')
    
    return log_spectrogram, spectrogram_image

def get_prediction(df_combined):
    model = load_model('model_1.h5')

    audio_data= df_combined
    audio_data_array = np.array(audio_data)
    audio_data_3d_array = audio_data_array[:,:,np.newaxis]

    m_predictions = model.predict(audio_data_3d_array[:1])

    prediction = Prediction(
        m_predictions[0][0],
        m_predictions[0][1], 
        m_predictions[0][2], 
        m_predictions[0][3], 
        m_predictions[0][4], 
        m_predictions[0][5], 
        m_predictions[0][6])

    return prediction


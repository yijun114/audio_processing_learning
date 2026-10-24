import numpy as np
import soundfile as sf
import librosa as lb
import matplotlib.pyplot as plt

data, samplerate = sf.read('testsound.wav')

#计算梅尔频谱图
S = lb.feature.melspectrogram(y=lb.to_mono(data.T), sr=samplerate)
 
#转换为分贝刻度
S_db_mel = lb.amplitude_to_db(S, ref=np.max)
 
#可视化梅尔频谱图
plt.figure(figsize=(10, 4))
lb.display.specshow(S_db_mel, sr=samplerate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.savefig('MelSpectrogram.png')
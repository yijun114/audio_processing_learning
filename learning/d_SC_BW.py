"""信号的频域特征
# 1. 加载信号
# 2. 获得信号的Spectral centroid和bandwidth
librosa.feature.spectral_centroid(y=waveform)
# 3. 获得信号的Spectral bandwidth
librosa.feature.spectral_bandwidth(y=waveform)
"""
import librosa
import numpy as np
from matplotlib import pyplot as plt

# 1. 加载信号
jazz_path = r"E:\Code\audio_data\jazz.wav"
jazz,sr = librosa.load(jazz_path,sr=None)

rock_path = r"E:\Code\audio_data\rock.wav"
rock,sr = librosa.load(rock_path,sr=None)

blues_path = r"E:\Code\audio_data\blues.wav"
blues,sr = librosa.load(blues_path,sr=None)

orchestra_path = r"E:\Code\audio_data\orchestra.wav"
orchestra,sr = librosa.load(orchestra_path,sr=None)
# 2. 获得信号的Spectral centroid
sc_jazz = librosa.feature.spectral_centroid(y=jazz,n_fft=1024).T[:,0]
sc_rock = librosa.feature.spectral_centroid(y=rock,n_fft=1024).T[:,0]
sc_blues = librosa.feature.spectral_centroid(y=blues,n_fft=1024).T[:,0]
sc_orchestra = librosa.feature.spectral_centroid(y=orchestra,n_fft=1024).T[:,0]

fig,aix = plt.subplots(2,2)
aix[0,0].plot(np.arange(0,len(sc_jazz)),sc_jazz,linewidth=1)
aix[0,0].set_title("Jazz")
aix[0,1].plot(np.arange(0,len(sc_rock)),sc_rock,linewidth=1)
aix[0,1].set_title("Rock")
aix[1,0].plot(np.arange(0,len(sc_blues)),sc_blues,linewidth=1)
aix[1,0].set_title("Blues")
aix[1,1].plot(np.arange(0,len(sc_orchestra)),sc_orchestra,linewidth=1)
aix[1,1].set_title("Orchestra")
fig.suptitle("Spectral Centroid")
plt.show()

# 3. 获得信号的Spectral bandwidth
sw_jazz = librosa.feature.spectral_bandwidth(y=jazz,n_fft=1024).T[:,0]
sw_rock = librosa.feature.spectral_bandwidth(y=rock,n_fft=1024).T[:,0]
sw_blues = librosa.feature.spectral_bandwidth(y=blues,n_fft=1024).T[:,0]
sw_orchestra = librosa.feature.spectral_bandwidth(y=orchestra,n_fft=1024).T[:,0]

figure,aix = plt.subplots(2,2)
aix[0,0].plot(np.arange(0,len(sw_jazz),),sw_jazz,linewidth=1)
aix[0,0].set_title("Jazz")
aix[0,1].plot(np.arange(0,len(sw_rock)),sw_rock,linewidth=1)
aix[0,1].set_title("Rock")
aix[1,0].plot(np.arange(0,len(sw_blues)),sw_blues,linewidth=1)
aix[1,0].set_title("Blues")
aix[1,1].plot(np.arange(0,len(sw_orchestra)),sw_orchestra,linewidth=1)
aix[1,1].set_title("Orchestra")
figure.suptitle("Spectral Bandwidth")
plt.show()
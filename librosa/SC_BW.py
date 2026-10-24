import librosa as lb
import numpy as np
from matplotlib import pyplot as plt

blues_path = r"C:\newwrold\my file\导师\硕士\声音处理\音频信号处理及深度学习教程\audio_data\blues.wav"
blues, blues_sr = lb.load(blues_path, sr=None)

jazz_path = r"C:\newwrold\my file\导师\硕士\声音处理\音频信号处理及深度学习教程\audio_data\jazz.wav"
jazz, jazz_sr = lb.load(jazz_path, sr=None)

rock_path = r"C:\newwrold\my file\导师\硕士\声音处理\音频信号处理及深度学习教程\audio_data\rock.wav"
rock, rock_sr = lb.load(rock_path, sr=None)

orchestra_path = r"C:\newwrold\my file\导师\硕士\声音处理\音频信号处理及深度学习教程\audio_data\orchestra.wav"
orchestra, orchestra_sr = lb.load(orchestra_path, sr=None)

sc_blues = lb.feature.spectral_centroid(y=blues, n_fft=1024).T[:,0]
sc_jazz = lb.feature.spectral_centroid(y=jazz, n_fft=1024).T[:,0]
sc_rock = lb.feature.spectral_centroid(y=rock, n_fft=1024).T[:,0]
sc_orchestra = lb.feature.spectral_centroid(y=orchestra, n_fft=1024).T[:,0]

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

sw_jazz = lb.feature.spectral_bandwidth(y=jazz,n_fft=1024).T[:,0]
sw_rock = lb.feature.spectral_bandwidth(y=rock,n_fft=1024).T[:,0]
sw_blues = lb.feature.spectral_bandwidth(y=blues,n_fft=1024).T[:,0]
sw_orchestra = lb.feature.spectral_bandwidth(y=orchestra,n_fft=1024).T[:,0]

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
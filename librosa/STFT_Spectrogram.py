import numpy as np
import soundfile as sf
import librosa as lb
import matplotlib.pyplot as plt

data, samplerate = sf.read('imcgx.wav')

#对音频做短时傅里叶变换，即按片段切分，每个片段进行快速傅里叶变换
#返回的是复数矩阵，形状为(频率格数, 帧数)，具体值含义为：第i个帧（即片段）上，k hz频率的强度是多少
#其中每一个复数 a + bj 都同时携带两种信息：幅度和相位
#stft支持多声道输入并保留声道维度，双声道会得到(声道数, 频率格数, 帧数)的三维数组
#specshow只能画二维的(频率格数, 帧数)，所以先用to_mono把左右声道平均成单声道
D = lb.stft(lb.to_mono(data.T))
 
#将复数谱转换为幅度谱
#np.abs(D)：取复数的模长，即幅度，也相当于丢弃了相位(因为可视化只关心"多强"，不关心"什么位置")
#因为人耳对声音强弱的感受不是线性的。声音能量可以差几千几万倍，但听起来可能只是“响一点”。分贝就是把这个范围压缩一下，更像人耳的感觉。
S_db = lb.amplitude_to_db(np.abs(D), ref=np.max)
 
#可视化频谱图
plt.figure(figsize=(10, 4))
lb.display.specshow(S_db, sr=samplerate, x_axis='time', y_axis='hz')
#在旁边加一个颜色条，说明什么颜色对应多少分贝
plt.colorbar(format='%+2.0f dB')
plt.title('spectrogram')
plt.tight_layout()
plt.savefig('spectrogram.png')
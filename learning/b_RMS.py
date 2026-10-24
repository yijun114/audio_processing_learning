"""信号的均方根值RootMeanSquareEnergy
# 0.预设环境
# 1.加载信号
# 2.定义函数RMS，功能：计算每一帧的均方根能量，
公式=该帧信号的平方和，取帧长的平均值后,开根号后
# 3.设置参数：每一帧长1024，以50%的重叠率分帧，调用该函数
# 4.绘制图像
# 5.利用librosa.feature.rms绘制信号的RMS
# 6.比较两者差异
"""
# 0.预设环境
import librosa
import numpy as np
from matplotlib import pyplot as plt
import librosa.display

# 1.加载信号
wave_path = r"E:\Code\audio_data\music_piano.wav"
waveform,sample_rate = librosa.load(wave_path,sr=None)

# 2.定义函数RMS，功能：计算每一帧的均方根能量，
def Calc_RMS(waveform,frame_length,hop_length):
    if len(waveform) % hop_length != 0:
        frame_num = int((len(waveform)-frame_length)/hop_length) + 1
        pad_num = frame_num * hop_length +frame_length - len(waveform)
        waveform = np.pad(waveform,pad_width=(0,pad_num),mode="wrap")
    frame_num = int((len(waveform)-frame_length)/hop_length) + 1
    waveform_rms = []
    for t in range(frame_num):
        current_frame = waveform[t*(frame_length-hop_length):t*(frame_length-hop_length)+frame_length]
        current_rms = np.sqrt(np.sum(current_frame**2) / frame_length)
        waveform_rms.append(current_rms)
    return np.array(waveform_rms)
# 3.设置参数：每一帧长1024，以50%的重叠率分帧，调用该函数
frame_size = 1024
hop_size = int(frame_size*0.5)
waveform_RMS = Calc_RMS(waveform,frame_size,hop_size)
# 4.绘制图像
frame_scale = np.arange(0,len(waveform_RMS),step=1)
time_scale = librosa.frames_to_time(frame_scale)
plt.figure(figsize=(20,10))
librosa.display.waveshow(waveform)
plt.plot(time_scale,waveform_RMS,color="r")
plt.title("Root-Mean-Square-Energy")
plt.show()
# 5.利用librosa.feature.rms绘制信号的RMS
waveform_RMS_librosa = librosa.feature.rms(y=waveform,frame_length=frame_size,hop_length=hop_size).T[1:,0]
plt.figure(figsize=(20,10))
librosa.display.waveshow(waveform)
plt.plot(time_scale,waveform_RMS_librosa,color="r")
plt.title("Root-Mean-Square-Energy_librosa")
plt.show()

bias = waveform_RMS_librosa - waveform_RMS
print(f"the bias is {bias}\n Congratulation!")

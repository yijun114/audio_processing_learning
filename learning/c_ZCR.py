"""信号的过零率ZCR
# 0.预设环境
# 1.加载信号
# 2.定义函数，功能：计算每一帧的过零率，
公式=该帧信号的平方和，开根号后，取帧长的平均值
np.sum(np.abs(np.sign(a)-np.sign(b)))/2/frame_length
# 3.设置参数：每一帧长1024，以50%的重叠率分帧，调用该函数
# 4.绘制图像
# 5.利用librosa.feature.
# 6.比较两者差异
"""
import librosa
import numpy as np
from matplotlib import pyplot as plt
import librosa.display

# 1.加载信号
wave_path = r"E:\Code\audio_data\speech_this_song.wav"
waveform,sampel_rate = librosa.load(wave_path,sr=None)

# 2.定义函数，功能：计算每一帧的过零率，
def Calc_ZCR(waveform,frame_length,hop_length):
    if len(waveform) % hop_length != 0:
        frame_num = int((len(waveform)-frame_length)/hop_length) + 1
        pad_num = frame_num*hop_length + frame_length - len(waveform)
        waveform = np.pad(waveform,pad_width=(0,pad_num),mode="wrap")
    frame_num = int((len(waveform)-frame_length)/hop_length) + 1
    waveform_zcr = []
    for t in range(frame_num):
        current_frame = waveform[(frame_length-hop_length)*t:(frame_length-hop_length)*t+frame_length]
        a = np.sign(current_frame[0:frame_length-1,])
        b = np.sign(current_frame[1:frame_length,])
        current_zcr = np.sum(np.abs(a-b))/2/frame_length
        waveform_zcr.append(current_zcr)
    return np.array(waveform_zcr)
# 3.设置参数：每一帧长1024，以50%的重叠率分帧，调用该函数
frame_size = 1024
hop_size = int(frame_size/2)
waveform_ZCR = Calc_ZCR(waveform,frame_size,hop_size)
# 4.绘制图像
frame_scale = np.arange(0,len(waveform_ZCR),step=1)
time_scale = librosa.frames_to_time(frame_scale,hop_length = hop_size)
plt.figure(figsize=(20,10))
plt.plot(time_scale,waveform_ZCR,color="r")
plt.title("Zero-Cross-rate")
plt.show()
# 5.利用librosa.feature.
waveform_zrc_librosa = librosa.feature.zero_crossing_rate(y=waveform,frame_length = frame_size,hop_length=hop_size).T[1:,0]
plt.figure(figsize=(20,10))
plt.plot(time_scale,waveform_zrc_librosa,color="r")
plt.title("Zero-Cross-rate-librosa")
plt.show()
bias = waveform_zrc_librosa - waveform_ZCR
print(f"the bias is {bias}\n Congratulation!")




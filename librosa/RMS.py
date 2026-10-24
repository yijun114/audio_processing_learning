import librosa as lb
import numpy as np
from matplotlib import pyplot as plt

waveform, samplerate = lb.load('imcgx.wav', sr=None)

def Calc_RMS(waveform, frame_size, hop_size):
    remainder = (len(waveform) - frame_size) % hop_size  # 超出上一个整数倍多少
    pad = (hop_size - remainder) % hop_size             # 还差多少到下一个整数倍；整除时修正为 0
    if pad > 0:
        waveform = np.pad(waveform, (0, pad), mode="constant")
    frame_num = (len(waveform) - frame_size) // hop_size + 1
    waveform_rms = []
    for t in range(frame_num):
        cur_frame = waveform[t*hop_size : t*hop_size+frame_size]
        cur_rms = np.sqrt(np.sum(cur_frame**2) / frame_size)
        waveform_rms.append(cur_rms)
    return waveform_rms

frame_size = 1024
hop_size = int(frame_size*0.5)
waveform_RMS = Calc_RMS(waveform,frame_size,hop_size)

# 4.绘制图像
frame_scale = np.arange(0,len(waveform_RMS),step=1)
time_scale = lb.frames_to_time(frame_scale)
plt.figure(figsize=(20,10))
lb.display.waveshow(waveform)
plt.plot(time_scale,waveform_RMS,color="r")
plt.title("Root-Mean-Square-Energy")
plt.show()
# 5.利用librosa.feature.rms绘制信号的RMS
waveform_RMS_librosa = lb.feature.rms(y=waveform,frame_length=frame_size,hop_length=hop_size).T[1:,0]
plt.figure(figsize=(20,10))
lb.display.waveshow(waveform)
plt.plot(time_scale,waveform_RMS_librosa,color="r")
plt.title("Root-Mean-Square-Energy_librosa")
plt.show()

bias = waveform_RMS_librosa - waveform_RMS
print(f"the bias is {bias}\n Congratulation!")

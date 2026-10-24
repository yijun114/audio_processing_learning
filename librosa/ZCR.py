import librosa as lb
import numpy as np
from matplotlib import pyplot as plt

waveform, samplerate = lb.load('imcgx.wav', sr=None)

def Calc_ZCR(waveform, frame_size, hop_size):
    remainder = (len(waveform) - frame_size) % hop_size  # 超出上一个整数倍多少
    pad = (hop_size - remainder) % hop_size             # 还差多少到下一个整数倍；整除时修正为 0
    if pad > 0:
        waveform = np.pad(waveform, (0, pad), mode="constant")
    frame_num = (len(waveform) - frame_size) // hop_size + 1
    waveform_zcr = []
    for t in range(frame_num):
        cur_frame = waveform[t*hop_size : t*hop_size+frame_size]
        a = np.sign(cur_frame[0 : frame_size-1])
        b = np.sign(cur_frame[1 : frame_size])
        cur_zcr = np.sum(np.abs(b - a))/2/frame_size
        waveform_zcr.append(cur_zcr)
    return waveform_zcr

# 3.设置参数：每一帧长1024，以50%的重叠率分帧，调用该函数
frame_size = 1024
hop_size = int(frame_size/2)
waveform_ZCR = Calc_ZCR(waveform,frame_size,hop_size)
# 4.绘制图像
frame_scale = np.arange(0,len(waveform_ZCR),step=1)
time_scale = lb.frames_to_time(frame_scale,hop_length = hop_size)
plt.figure(figsize=(20,10))
plt.plot(time_scale,waveform_ZCR,color="r")
plt.title("Zero-Cross-rate")
plt.show()
# 5.利用librosa.feature.
waveform_zrc_librosa = lb.feature.zero_crossing_rate(y=waveform,frame_length = frame_size,hop_length=hop_size).T[1:,0]
plt.figure(figsize=(20,10))
plt.plot(time_scale,waveform_zrc_librosa,color="r")
plt.title("Zero-Cross-rate-librosa")
plt.show()
bias = waveform_zrc_librosa - waveform_ZCR
print(f"the bias is {bias}\n Congratulation!")
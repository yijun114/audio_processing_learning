import librosa as lb
import numpy as np
from matplotlib import pyplot as plt

waveform, samplerate = lb.load('imcgx.wav', sr=None)

def Calc_Amplitude_Envelope(waveform, frame_size, hop_size):
    #先判断要不要补帧
    if (len(waveform)-frame_size) % hop_size != 0:
        frame_num = int((len(waveform)-frame_size)/hop_size) + 1
        pad_num = frame_num * hop_size + frame_size - len(waveform)
        waveform = np.pad(waveform, pad_width=(0, pad_num), mode='constant')
    frame_num = int((len(waveform)-frame_size)/hop_size) + 1
    waveform_ae_plus = []
    waveform_ae_minus = []
    for t in range(frame_num):
        cur_frame = waveform[t*hop_size : t*hop_size+frame_size]
        cur_ae_plus = max(cur_frame)
        cur_ae_minus = min(cur_frame)
        waveform_ae_plus.append(cur_ae_plus)
        waveform_ae_minus.append(cur_ae_minus)
    return waveform_ae_plus, waveform_ae_minus

def Calc_Amplitude_Envelope_2(waveform, frame_size, hop_size):
    # 补零：保证 (len - frame_length) 能被 hop_length 整除，最后一帧完整
    remainder = (len(waveform) - frame_size) % hop_size  # 超出上一个整数倍多少
    pad = (hop_size - remainder) % hop_size             # 还差多少到下一个整数倍；整除时修正为 0
    if pad > 0:
        waveform = np.pad(waveform, (0, pad), mode="constant")
    frame_num = (len(waveform) - frame_size) // hop_size + 1
    waveform_ae = []
    for t in range(frame_num):
        start = t * hop_size                      # 关键：步进必须是 hop_length
        current_frame = waveform[start : start + frame_size]
        current_ae = max(np.abs(current_frame))     # 包络取绝对值
        waveform_ae.append(current_ae)
    return np.array(waveform_ae)

frame_size = 1024
hop_size = int(frame_size*0.5)
waveform_AE_plus, waveform_AE_minus = Calc_Amplitude_Envelope(waveform=waveform, frame_size=frame_size, hop_size=hop_size)

frame_scale = np.arange(0,len(waveform_AE_plus))
time_scale = lb.frames_to_time(frame_scale,hop_length=hop_size)
plt.figure(figsize=(20,10))
lb.display.waveshow(waveform)
plt.plot(time_scale,waveform_AE_plus,color="r")
plt.plot(time_scale,waveform_AE_minus,color="r")
plt.title("Amplitude_Envelope")
plt.show()
print("Congratulation!you finish it!")
import librosa as lb
import soundfile as sf

data, samplerate = sf.read('testsound.wav')
print('原始音频采样率：', samplerate)

#将音频重采样到16000Hz
#注意lb.resample默认沿最后一维(axis=-1)重采样，而sf.read返回的形状是(帧数, 声道数)，
#不指定axis=0的话会把只有2个采样的声道轴当成时间轴，结果形状不变、时长变成3倍
data_16k = lb.resample(data, orig_sr=samplerate, target_sr=16000, axis=0)

sf.write('resrSound.wav', data_16k, 16000)

data2, samplerate2 = sf.read('resrSound.wav')
print('重采样后采样率：', samplerate2)

#时间拉伸，速度变为原来的1.5倍
#time_stretch沿最后一维(时间轴)处理，而sf.read返回(帧数, 声道数)，
#所以先转置成(声道数, 帧数)，处理完再转回来
stretched_data = lb.effects.time_stretch(data.T, rate=1.5).T
sf.write('fasterSound.wav', stretched_data, samplerate)

#音调偏移，上移2个半音
#pitch_shift同样是沿最后一维处理，也要先转置
shifted_data = lb.effects.pitch_shift(data.T, sr=samplerate, n_steps=2).T
sf.write('higherSound.wav', shifted_data, samplerate)

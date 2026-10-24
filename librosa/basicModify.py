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

#修剪静音部分
#trim函数有两个返回值，第一个是处理后的声音形状数组
#第二个是一个二元素数组index，其标记的是"保留的这段音频，对应原数组的第几个采样点到第几个采样点"
#下划线“_”是python的语法，表示一个普通变量，其中的含义是“我接收这个变量，但我不使用这个变量”
#由于这个测试音频不存在静音的部分，故而执行后没有变化
data_trimmed, _ = lb.effects.trim(data.T)
sf.write('removeMuteSound.wav', data_trimmed.T, samplerate)
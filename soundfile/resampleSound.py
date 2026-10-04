import soundfile as sf
from scipy.signal import resample_poly

print('开始处理')

data, samplerate = sf.read('testsound.wav')

print('原采样率:', samplerate)

new_rate = 2000

#转换为新的采样率:scipy 按 新采样率/旧采样率 的比例重新计算采样点
#采样率低了以后，听感很闷
converted_data = resample_poly(data, new_rate, samplerate)

sf.write('convertedsound.wav', converted_data, new_rate)

print('转换完成!')
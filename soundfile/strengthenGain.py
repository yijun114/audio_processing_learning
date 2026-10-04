import soundfile as sf
import numpy as np

data, samplerate = sf.read('testsound.wav')

#定义增益因子，并进行应用
gain_factor = 2.0
data_gained = data * gain_factor

#clip函数可以把数组内的最大值拉低，最小值拉高，让所有值限定在范围内
data_gained = np.clip(data_gained, -1.0, 1.0)

sf.write('gainedSound.wav', data_gained, samplerate)
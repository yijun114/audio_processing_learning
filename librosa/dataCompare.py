import librosa as lb
import soundfile as sf

print("librosa版本:", lb.__version__)

data, samplerate = lb.load('testsound.wav')

#读出的数据形状为(42336, )，采样率为22050，而非soundfile的(92159, 2)和48000
#因为两个库定位不同，sf主要是负责高效读入写入，输出的是音频原本的数据
#而lb面向音频分析，而分析通常不需要48000Hz那么高的采样率，也不需要双声道，22050Hz是音频分析领域的传统标准
#所以lb输出的是重采样后的数据，计算公式是92159 × (22050/48000) ≈ 42336
print("音频数据形状:", data.shape)
print("采样率:", samplerate)
print('------')

#load函数加上这两个参数，输出的就会是原本的音频信息了
data2, samplerate2 = lb.load('testsound.wav', sr=None, mono=False)

#注意lb库输出的音频数据形状格式和sf库相反，sf库是先帧数再声道数，sf库则是先声道数再帧数
print("音频数据形状:", data2.shape)
print("采样率:", samplerate2)
print('------')

#补充sf库的输出，方便对比
data_sf, samplerate_sf = sf.read('testsound.wav')

print("音频数据形状:", data_sf.shape)
print("采样率:", samplerate_sf)
import soundfile as sf

#加载原始音频文件
data, samplerate = sf.read('testsound.wav')
#返回：帧数+声道数(单声道则无)
print('音频数据形状:', data.shape)
#音频本身只是一串数字，它没有“时间”这个概念，它只能每隔极短的时间间隔测量一次声音的大小，记下一个数字。这个测量频率就是采样率
print('采样率:', samplerate)

#给出起始的时间节点
start_time = 0.5
end_time = 1.0

#根据起始的时间节点和采样率（测量的时间间隔），计算目标“数字”的位置，进行剪辑
#强转int是因为采样点只有第x个，没有小数位
start_index = int(start_time * samplerate)
end_index = int(end_time * samplerate)

clipped_data = data[start_index:end_index]

sf.write('clippedsound.wav', clipped_data, samplerate)
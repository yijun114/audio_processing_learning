import librosa as lb
import matplotlib.pyplot as plt

#python有两种传参方式：按顺序传和按名字传，此处为后者
#如果按前者，也可以写作lb.load('testsound.wav', None, False)
#但load函数定义中强制规定，只能按名字传，所以按顺序传的写法是非法的
data, samplerate = lb.load('testsound.wav', sr=None, mono=False)

#定义画布大小
plt.figure(figsize=(10, 4))
#lb库底层会调用matpilot库进行绘画
lb.display.waveshow(data, sr=samplerate)
#设置标题、横轴、纵轴
plt.title('Audio waveform diagram')
plt.xlabel('time/s')
plt.ylabel('Amplitude')
#自动调整边距， 防止标签被裁掉
plt.tight_layout()
#存储为图片
plt.savefig('wave.png')

print('绘制已完成')
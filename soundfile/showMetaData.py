import soundfile as sf

metadata = sf.info('testsound.wav')
print('音频元数据如下：', metadata)
#输出的metadata中subtype是Signed 16 bit PCM，即带符号16位整型PCM，但是sf.read()得到的data数值却是在-1.0到1.0之间的浮点数
#这是因为sf.read()方法得到的数值是经过归一化的，Signed 16 bit PCM是音频的真实数值，read()的结果是归一化后的64位浮点数类型
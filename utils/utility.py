# coding:utf-8
# Date : 2019.07.23
# author: phecda
#
#
# ********************************
import struct
import scipy.signal
import scipy.io.wavfile as wav
from python_speech_features import mfcc, delta, fbank



def genHeadInfo(sampleRate, bits, sampleNum, channels):
    '''
    # TODO 生成头信息
    :param sampleRate:  采样率，int
    :param bits:        每个采样点的位数，int
    :param sampleNum:   音频字节数，int
    :return:            音频头文件，byte
    '''
    rHeadInfo = b'\x52\x49\x46\x46'                               # RIFF
    fileLength = struct.pack('i', sampleNum + 36)                 # sampleNum = 总采样点数 × 2  = 秒数 × 采样率 × 2
    rHeadInfo += fileLength                                       # 去掉RIFF 部分 剩余的音频文件总长度  32036 -> b'$}\x00\x00'
    rHeadInfo += b'\x57\x41\x56\x45\x66\x6D\x74\x20'              # b'WAVEfmt '
    rHeadInfo += struct.pack('i', bits)                           # 每个采样的位数 16 b'\x10\x00\x00\x00'
    rHeadInfo += b'\x01\x00'                                      # 音频数据的编码方式 1
    rHeadInfo += struct.pack('h', channels)                       # 声道数 1 ， short型数据  1->b'\x01\x00',2->b'\x02\x00'
    rHeadInfo += struct.pack('i', sampleRate)                     # 采样率 16000 -> b'\x80>\x00\x00'
    rHeadInfo += struct.pack('i', int(sampleRate * bits / 8))     # 音频传输速率  32000 = 16000*16/8  -> b'\x00}\x00\x00'
    rHeadInfo += b'\x02\x00\x10\x00'                              # [2,16] 作用未知，默认不需改动
    rHeadInfo += b'\x64\x61\x74\x61'                              # data
    rHeadInfo += struct.pack('i', sampleNum)                      # 不包括头文件的音频长度 32000 \x00}\x00\x00
    return rHeadInfo


def gen_pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):
    M = scipy.signal.lfilter([s], [1, s - 1], E)
    smooth = (eps + M)**(-alpha)
    return (E * smooth + delta)**r - delta**r

def gen_feature(wvfile):
    (rate, sig) = wav.read(wvfile)
    # For multi-channel .wav file, use the first channel.
    if len(sig.shape) == 2:
        sig = sig[:, 0]
    # 40-channel Mel-filterbank, 25ms windowsing, 10ms frame shift.
    (fb_feature, total_energy) = fbank(sig, rate, nfft=2048, nfilt=40)
    pcen_feature = gen_pcen(fb_feature)
    return pcen_feature
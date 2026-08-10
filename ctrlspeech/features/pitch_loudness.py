import numpy as np
import librosa
import pyworld as pw
from scipy.signal import zpk2sos, sosfilt, bilinear_zpk


# ========= 全局常量 =========
f0_bin     = 128         # 0..127
f0_max     = 650.0
f0_min     = 65.0

loudness_bin = 64        # 0..63
loudness_min = 1e-10     # 避免 log(0)
loudness_db_floor = -60.0  # dB 下限 (实际语音很少低于此值)
loudness_db_max = 0.0      # dB 上限
# 分辨率: 60 dB / 64 bins ≈ 0.94 dB/bin (接近人耳 JND)


def f0_to_coarse(f0):
    """将 F0 (Hz) 转换为量化的 pitch 值 (0-127)"""
    above_max = f0 > f0_max

    f0_mel     = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    # 把 >0 的 voiced 帧线性映射到 1..126
    voiced = f0_mel > 0
    f0_mel[voiced] = (
        (f0_mel[voiced] - f0_mel_min)
        * (f0_bin - 2)
        / (f0_mel_max - f0_mel_min)
        + 1
    )

    f0_mel[f0_mel < 0]         = 1
    f0_mel[f0_mel > f0_bin-1]  = f0_bin - 1
    f0_mel[above_max]          = f0_bin - 1     # 超范围 → 127

    f0_coarse = np.rint(f0_mel).astype(np.int32)
    assert f0_coarse.max() < f0_bin and f0_coarse.min() >= 0
    return f0_coarse


def loudness_to_coarse(loudness_db, db_max=loudness_db_max, db_floor=loudness_db_floor):
    """
    把 dB 响度量化到 0..63
    - db_max: dB 上限 (默认 0 dB)
    - db_floor: dB 下限 (默认 -60 dB)
    - 分辨率: 约 0.94 dB/bin
    """
    assert db_floor < db_max

    # clip 到范围内
    loudness_clipped = np.clip(loudness_db, db_floor, db_max)

    # 归一化后量化
    loudness_norm = (loudness_clipped - db_floor) / (db_max - db_floor)  # 0..1
    loudness_coarse = np.round(loudness_norm * (loudness_bin - 1)).astype(np.int32)

    assert loudness_coarse.max() < loudness_bin and loudness_coarse.min() >= 0
    return loudness_coarse


# ========= A-weighting 滤波器 (IEC 61672-1) =========
def a_weighting_sos(sr):
    """
    生成 A-weighting 滤波器的 SOS (Second-Order Sections) 系数
    基于 IEC 61672-1 标准
    """
    # A-weighting 的特征频率
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    
    # 模拟域零点和极点
    zeros = [0, 0, 0, 0]
    poles = [
        -2 * np.pi * f1,
        -2 * np.pi * f1,
        -2 * np.pi * f2,
        -2 * np.pi * f3,
        -2 * np.pi * f4,
        -2 * np.pi * f4
    ]
    
    # A-weighting 增益 (使 1kHz 处增益为 0 dB)
    # K = (2*pi*f4)^2 * (2*pi*f1)^2 / A1000
    k = (2 * np.pi * f4)**2 * (2 * np.pi * f1)**2
    
    # 双线性变换到数字域
    z, p, k = bilinear_zpk(zeros, poles, k, sr)
    
    # 归一化使 1kHz 增益为 0 dB
    # 计算 1kHz 处的增益
    w = 2 * np.pi * 1000 / sr
    h_1k = k * np.prod(np.exp(1j * w) - z) / np.prod(np.exp(1j * w) - p)
    k = k / np.abs(h_1k)
    
    # 转换为 SOS 格式 (更稳定)
    sos = zpk2sos(z, p, k)
    return sos


def apply_a_weighting(wav, sr):
    """对音频应用 A-weighting 滤波"""
    sos = a_weighting_sos(sr)
    return sosfilt(sos, wav)


# ========= 核心提取函数 =========
def get_pitch(wav, sr=16000, hop_length=320):
    """
    逐帧提取 pitch (F0)
    
    返回:
        f0: 原始 F0 值 (Hz)，unvoiced 帧为 0
        f0_coarse: 量化后的 pitch (0-127)
    """
    _f0, t = pw.dio(wav.astype(np.double), sr, frame_period=hop_length / sr * 1000)
    f0 = pw.stonemask(wav.astype(np.double), _f0, t, sr)
    return f0, f0_to_coarse(f0)


def get_loudness(
    wav, sr=16000, hop_length=320, 
    quantize=True, db_max=loudness_db_max, db_floor=loudness_db_floor,
    use_a_weighting=True
):
    """
    逐帧提取响度 (Loudness)
    
    与 energy 的区别:
    - Loudness 使用 A-weighting 滤波器模拟人耳的频率响应
    - 直接返回 dB 值，更符合人类对响度的感知
    
    参数:
        wav: 音频波形 (numpy array)
        sr: 采样率
        hop_length: 帧移
        quantize: 是否量化
        db_max: dB 上限
        db_floor: dB 下限
        use_a_weighting: 是否使用 A-weighting
    
    返回:
        loudness_db: 每帧的响度 (dB)
        loudness_coarse: 量化后的响度 (0-63) [如果 quantize=True]
    """
    frame_len = hop_length * 2
    
    if use_a_weighting:
        # 应用 A-weighting 滤波
        wav_weighted = apply_a_weighting(wav, sr)
    else:
        wav_weighted = wav
    
    # 计算逐帧 RMS
    rms = librosa.feature.rms(y=wav_weighted, frame_length=frame_len, hop_length=hop_length)[0]
    
    # 转换为 dB
    rms_safe = np.maximum(rms, loudness_min)
    loudness_db = 20 * np.log10(rms_safe)
    
    # clip 到范围
    loudness_db = np.clip(loudness_db, db_floor, db_max)
    
    if quantize:
        loudness_coarse = loudness_to_coarse(loudness_db, db_max, db_floor)
        return loudness_db, loudness_coarse
    return loudness_db


def get_pitch_and_loudness(wav, sr=16000, hop_length=320, quantize=True):
    """
    一次性逐帧提取 pitch 和 loudness
    
    返回:
        f0: 原始 F0 (Hz)
        f0_coarse: 量化后的 pitch (0-127)
        loudness_db: 响度 (dB)
        loudness_coarse: 量化后的响度 (0-63) [如果 quantize=True]
    """
    f0, f0_coarse = get_pitch(wav, sr, hop_length)
    
    if quantize:
        loudness_db, loudness_coarse = get_loudness(wav, sr, hop_length, quantize=True)
        
        # 确保帧数对齐
        min_len = min(len(f0), len(loudness_db))
        f0, f0_coarse = f0[:min_len], f0_coarse[:min_len]
        loudness_db, loudness_coarse = loudness_db[:min_len], loudness_coarse[:min_len]
        
        return f0, f0_coarse, loudness_db, loudness_coarse
    else:
        loudness_db = get_loudness(wav, sr, hop_length, quantize=False)
        
        min_len = min(len(f0), len(loudness_db))
        f0, f0_coarse = f0[:min_len], f0_coarse[:min_len]
        loudness_db = loudness_db[:min_len]
        
        return f0, f0_coarse, loudness_db
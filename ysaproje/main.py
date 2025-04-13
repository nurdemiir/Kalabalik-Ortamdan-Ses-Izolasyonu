import nemo.collections.asr as nemo_asr
import torch
import torchaudio
import os
import sys
import logging
import warnings
try:
    from huggingface_hub import ModelFilter
except ImportError:
    try:
        from huggingface_hub.utils import ModelFilter
    except ImportError:
        class ModelFilter:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
        warnings.warn("Using dummy ModelFilter implementation", RuntimeWarning)

os.environ['NEMO_SKIP_NLP_IMPORTS'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import noisereduce as nr
import soundfile as sf
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from pydub import AudioSegment, effects
import librosa
import webrtcvad
import whisper
from sklearn.metrics.pairwise import cosine_similarity

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('audio_isolation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# KONFİGÜRASYON SINIFI
@dataclass
class AudioConfig:
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    BIT_DEPTH: str = 'PCM_24'
    FRAME_DURATION_MS: int = 30
    
@dataclass
class ModelConfig:
    WHISPER_MODEL: str = "large-v3"
    SPEAKER_MODEL: str = "nvidia/speakerverification_en_titanet_large"
    
@dataclass
class ProcessingConfig:
    VAD_AGGRESSIVENESS: int = 2  # Daha az agresif VAD
    MIN_SPEAKER_DURATION: float = 0.5
    SIMILARITY_THRESHOLD: float = 0.75
    MIN_AUDIO_LENGTH: float = 0.2
    FADE_DURATION: float = 0.05
    NOISE_REDUCTION_PROP: float = 0.15  # Hafif gürültü azaltma
    DYNAMIC_RANGE_THRESHOLD: float = -5.0  # Minimum sıkıştırma
    DYNAMIC_RANGE_RATIO: float = 1.1  # Neredeyse hiç sıkıştırma
    LOW_PASS_FREQ: int = 14000  # Geniş bant
    HIGH_PASS_FREQ: int = 40  # Düşük kesim
    TRIM_SILENCE_THRESHOLD: float = 35.0  # Agresif kırpma

@dataclass
class PathConfig:
    REFERENCE_SPEAKER_AUDIO: str = "sadece_konusmaci.wav"
    MIXED_AUDIO: str = "kalabalik_ses.wav"
    OUTPUT_PATH: str = "output_enhad1.wav"
    TEMP_DIR: str = "temp_audio"

class Config:
    AUDIO = AudioConfig()
    MODEL = ModelConfig()
    PROCESSING = ProcessingConfig()
    PATHS = PathConfig()
    
    @property
    def DEVICE(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YARDIMCI FONKSİYONLAR
class AudioUtils:
    @staticmethod
    def is_speech(audio_chunk: np.ndarray, sample_rate: int, vad: webrtcvad.Vad) -> bool:
        frame_duration = Config.AUDIO.FRAME_DURATION_MS
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        frame_size = int(sample_rate * frame_duration / 1000)
        
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]
            if len(frame) < frame_size:
                continue
            if vad.is_speech(frame.tobytes(), sample_rate):
                return True
        return False
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        return audio / max_val if max_val > 0 else audio
    
    @staticmethod
    def apply_fade(audio: np.ndarray, fade_duration: float, sample_rate: int) -> np.ndarray:
        fade_samples = int(fade_duration * sample_rate)
        if len(audio) > 2 * fade_samples:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        return audio
    
    @staticmethod
    def smooth_transitions(audio: np.ndarray, sample_rate: int, window_size: float = 0.1) -> np.ndarray:
        window_samples = int(window_size * sample_rate)
        if len(audio) > window_samples:
            window = np.hanning(2 * window_samples)
            audio[:window_samples] *= window[:window_samples]
            audio[-window_samples:] *= window[window_samples:]
        return audio
    
    @staticmethod
    def load_audio(file_path: str, target_sample_rate: int) -> Tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(file_path)
        if sr != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, target_sample_rate
    
    @staticmethod
    def create_temp_dir() -> None:
        if not os.path.exists(Config.PATHS.TEMP_DIR):
            os.makedirs(Config.PATHS.TEMP_DIR)

# SPEAKER EMBEDDING
class SpeakerEmbeddingExtractor:
    def __init__(self):
        self.model = EncDecSpeakerLabelModel.from_pretrained(Config.MODEL.SPEAKER_MODEL)
        self.model.eval().to(Config().DEVICE)
    
    def extract(self, audio_path: str) -> np.ndarray:
        logger.info("Konuşmacı ses imzası oluşturuluyor...")
        
        waveform, _ = AudioUtils.load_audio(audio_path, Config.AUDIO.SAMPLE_RATE)
        
        segment_length = 5 * Config.AUDIO.SAMPLE_RATE
        embeddings = []
        
        for i in range(0, waveform.shape[1], segment_length):
            segment = waveform[:, i:i+segment_length]
            if segment.shape[1] < segment_length:
                continue
                
            with torch.no_grad():
                _, embedding = self.model.forward(
                    input_signal=segment.to(Config().DEVICE),
                    input_signal_length=torch.tensor([segment.shape[1]]).to(Config().DEVICE)
                )
            embeddings.append(embedding.squeeze().cpu().numpy())
        
        return np.mean(embeddings, axis=0)

# SES İZOLASYON MOTORU
class VoiceIsolationEngine:
    def __init__(self):
        self.whisper_model = whisper.load_model(Config.MODEL.WHISPER_MODEL)
        self.speaker_model = SpeakerEmbeddingExtractor()
        self.vad = webrtcvad.Vad(Config.PROCESSING.VAD_AGGRESSIVENESS)
    
    def isolate_speaker(self, mixed_path: str, reference_embedding: np.ndarray) -> np.ndarray:
        logger.info("Hedef konuşmacı izolasyonu başlatıldı...")
        
        result = self.whisper_model.transcribe(mixed_path, word_timestamps=True)
        waveform, _ = AudioUtils.load_audio(mixed_path, Config.AUDIO.SAMPLE_RATE)
        
        target_segments = []
        
        for segment in result["segments"]:
            start_sample = int(segment["start"] * Config.AUDIO.SAMPLE_RATE)
            end_sample = int(segment["end"] * Config.AUDIO.SAMPLE_RATE)
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[1], end_sample)
            
            segment_audio = waveform[:, start_sample:end_sample]
            
            if segment_audio.shape[1] < Config.PROCESSING.MIN_AUDIO_LENGTH * Config.AUDIO.SAMPLE_RATE:
                continue
                
            if self._is_target_speaker(segment_audio, reference_embedding):
                segment_np = segment_audio.numpy()[0]
                # Segment bazında sessiz kısımları kırp
                segment_np, _ = librosa.effects.trim(
                    segment_np,
                    top_db=Config.PROCESSING.TRIM_SILENCE_THRESHOLD,
                    frame_length=512,
                    hop_length=128
                )
                if len(segment_np) > Config.PROCESSING.MIN_AUDIO_LENGTH * Config.AUDIO.SAMPLE_RATE:
                    segment_np = AudioUtils.apply_fade(segment_np, Config.PROCESSING.FADE_DURATION, Config.AUDIO.SAMPLE_RATE)
                    target_segments.append(segment_np)
        
        if not target_segments:
            raise ValueError("Hedef konuşmacı bulunamadı!")
        
        # Overlap-add ile birleştirme
        overlap_samples = int(0.05 * Config.AUDIO.SAMPLE_RATE)  # 50ms overlap
        total_length = sum(len(seg) for seg in target_segments) - (len(target_segments) - 1) * overlap_samples
        isolated_audio = np.zeros(total_length)
        current_pos = 0
        
        for seg in target_segments:
            seg_len = len(seg)
            if current_pos + seg_len > len(isolated_audio):
                seg = seg[:len(isolated_audio) - current_pos]
                seg_len = len(seg)
            isolated_audio[current_pos:current_pos + seg_len] += seg
            current_pos += seg_len - overlap_samples
        
        # Son kırpma
        isolated_audio, _ = librosa.effects.trim(
            isolated_audio,
            top_db=Config.PROCESSING.TRIM_SILENCE_THRESHOLD,
            frame_length=512,
            hop_length=128
        )
        
        return AudioUtils.normalize_audio(isolated_audio)
    
    def _is_target_speaker(self, audio_segment: torch.Tensor, reference_embedding: np.ndarray) -> bool:
        with torch.no_grad():
            _, segment_embedding = self.speaker_model.model.forward(
                input_signal=audio_segment.to(Config().DEVICE),
                input_signal_length=torch.tensor([audio_segment.shape[1]]).to(Config().DEVICE)
            )
            segment_embedding = segment_embedding.squeeze().cpu().numpy()
        
        similarity = cosine_similarity(
            reference_embedding.reshape(1, -1),
            segment_embedding.reshape(1, -1)
        )[0][0]
        
        return similarity > Config.PROCESSING.SIMILARITY_THRESHOLD

# SES İYİLEŞTİRME
class AudioEnhancer:
    def __init__(self):
        pass
    
    @staticmethod
    def reduce_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=False,
            prop_decrease=Config.PROCESSING.NOISE_REDUCTION_PROP,
            n_fft=4096,  # Daha büyük FFT ile doğal tonlar korunur
            win_length=1024,
            time_constant_s=1.0
        )
    
    @staticmethod
    def spectral_enhancement(audio: np.ndarray) -> np.ndarray:
        stft = librosa.stft(audio, n_fft=4096, hop_length=1024)
        mag, phase = librosa.magphase(stft)
        freqs = librosa.fft_frequencies(sr=Config.AUDIO.SAMPLE_RATE, n_fft=4096)
        # 1-8 kHz aralığını hafifçe güçlendir
        boost_mask = (freqs >= 1000) & (freqs <= 8000)
        mag[boost_mask, :] *= 1.5  # %50 güçlendirme
        return librosa.istft(mag * phase, hop_length=1024, length=len(audio))
    
    @staticmethod
    def post_process(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_segment = AudioSegment(
            (audio * 32767).astype(np.int16).tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        audio_segment = audio_segment.low_pass_filter(Config.PROCESSING.LOW_PASS_FREQ).high_pass_filter(Config.PROCESSING.HIGH_PASS_FREQ)
        
        audio_segment = effects.compress_dynamic_range(
            audio_segment,
            threshold=Config.PROCESSING.DYNAMIC_RANGE_THRESHOLD,
            ratio=Config.PROCESSING.DYNAMIC_RANGE_RATIO,
            attack=20.0,
            release=200.0
        )
        
        audio_segment = effects.normalize(audio_segment, headroom=2.0)
        return np.array(audio_segment.get_array_of_samples()) / 32767.0
    
    # HiFi-GAN ile ses restorasyonu 
    def enhance_with_vocoder(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        # Bu fonksiyonu kullanmak için HiFi-GAN kurulu olmalı ve __init__ içindeki vocoder aktif edilmeli
        spec = librosa.stft(audio, n_fft=4096, hop_length=1024)
        mag, phase = librosa.magphase(spec)
        
        audio_tensor = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).to(Config().DEVICE)
        enhanced_audio = self.vocoder(audio_tensor).squeeze().cpu().numpy()
        
        if len(enhanced_audio) > len(audio):
            enhanced_audio = enhanced_audio[:len(audio)]
        elif len(enhanced_audio) < len(audio):
            enhanced_audio = np.pad(enhanced_audio, (0, len(audio) - len(enhanced_audio)))
        
        return AudioUtils.normalize_audio(enhanced_audio)
    
    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        reduced_noise = self.reduce_noise(audio, sample_rate)
        enhanced = self.spectral_enhancement(reduced_noise)
        post_processed = self.post_process(enhanced, sample_rate)
        # HiFi-GAN kullanmak istersen bu satırı aktif et
        # return self.enhance_with_vocoder(post_processed, sample_rate)
        return post_processed

# ANA İŞLEM
class AudioIsolationSystem:
    def __init__(self):
        self.logger = logger
        self.embedding_extractor = SpeakerEmbeddingExtractor()
        self.isolation_engine = VoiceIsolationEngine()
        self.audio_enhancer = AudioEnhancer()
        self._initialize()
    
    def _initialize(self) -> None:
        AudioUtils.create_temp_dir()
        torch.manual_seed(42)
        
    def _validate_inputs(self) -> None:
        if not os.path.exists(Config.PATHS.REFERENCE_SPEAKER_AUDIO):
            raise FileNotFoundError(f"Referans ses dosyası bulunamadı: {Config.PATHS.REFERENCE_SPEAKER_AUDIO}")
        if not os.path.exists(Config.PATHS.MIXED_AUDIO):
            raise FileNotFoundError(f"Karışık ses dosyası bulunamadı: {Config.PATHS.MIXED_AUDIO}")
        
        if not Config.PATHS.REFERENCE_SPEAKER_AUDIO.endswith('.wav'):
            raise ValueError("Referans ses dosyası .wav formatında olmalıdır")
        if not Config.PATHS.MIXED_AUDIO.endswith('.wav'):
            raise ValueError("Karışık ses dosyası .wav formatında olmalıdır")
    
    def _save_output(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            sf.write(
                Config.PATHS.OUTPUT_PATH,
                audio,
                sample_rate,
                subtype=Config.AUDIO.BIT_DEPTH
            )
            
            original_duration = librosa.get_duration(filename=Config.PATHS.MIXED_AUDIO)
            output_duration = len(audio) / sample_rate
            self.logger.info(f"BAŞARILI! Çıktı kaydedildi: {Config.PATHS.OUTPUT_PATH}")
            self.logger.info(f"Orijinal süre: {original_duration:.2f}s | İzole edilmiş süre: {output_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Çıktı kaydedilirken hata oluştu: {str(e)}")
            raise

    def _cleanup_temp_files(self) -> None:
        try:
            for filename in os.listdir(Config.PATHS.TEMP_DIR):
                file_path = os.path.join(Config.PATHS.TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    self.logger.warning(f"Geçici dosya silinemedi {file_path}: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Geçici dosya temizleme hatası: {str(e)}")

    def run(self) -> None:
        try:
            self.logger.info("Gelişmiş Ses İzolasyon Sistemi Başlatıldı")
            self.logger.info(f"Kullanılan cihaz: {Config().DEVICE}")
            
            self._validate_inputs()
            
            reference_embedding = self.embedding_extractor.extract(Config.PATHS.REFERENCE_SPEAKER_AUDIO)
            self.logger.info("Referans konuşmacı embedding'i başarıyla oluşturuldu")
            
            isolated = self.isolation_engine.isolate_speaker(Config.PATHS.MIXED_AUDIO, reference_embedding)
            self.logger.info(f"Konuşmacı izolasyonu tamamlandı. İzole edilen ses uzunluğu: {len(isolated)/Config.AUDIO.SAMPLE_RATE:.2f}s")
            
            enhanced = self.audio_enhancer.enhance(isolated, Config.AUDIO.SAMPLE_RATE)
            self.logger.info("Ses iyileştirme işlemleri tamamlandı")
            
            self._save_output(enhanced, Config.AUDIO.SAMPLE_RATE)
            
            self._cleanup_temp_files()
            
            self.logger.info("Tüm işlemler başarıyla tamamlandı")
            
        except KeyboardInterrupt:
            self.logger.info("İşlem kullanıcı tarafından durduruldu")
            self._cleanup_temp_files()
            sys.exit(0)
            
        except Exception as e:
            self.logger.error(f"KRİTİK HATA: {str(e)}", exc_info=True)
            self._cleanup_temp_files()
            sys.exit(1)

if __name__ == "__main__":
    system = AudioIsolationSystem()
    system.run()
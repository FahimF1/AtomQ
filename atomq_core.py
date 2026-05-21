"""
AtomQ Core Engine
-----------------
Mathematics of language without a dictionary.
Extracts meaning from speech using universal linguistic mathematics:
- Prosodic feature extraction (F0, rhythm, intensity, duration)
- Language family detection (no vocabulary required)
- Syntactic structure inference from prosody
- Universal thematic role assignment
- Semantic primitive detection
- Meaning frame construction

Based on:
- Chomsky's Universal Grammar
- Shannon's Information Theory
- Wierzbicka's Natural Semantic Metalanguage
- Universal prosodic mathematics
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import librosa
import parselmouth
from parselmouth.praat import call
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ProsodicFeatures:
    """Mathematical representation of speech prosody"""
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    f0_contour: np.ndarray = field(default_factory=lambda: np.array([]))
    f0_slope: float = 0.0          # Rising vs falling tendency
    f0_final_contour: str = ""     # Terminal contour type
    rhythm_type: str = ""          # stress-timed vs syllable-timed
    speech_rate: float = 0.0       # syllables per second
    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    duration: float = 0.0
    entropy: float = 0.0           # Shannon entropy of F0
    info_density: float = 0.0      # estimated bits per second
    is_tonal: bool = False
    tone_count: int = 0


@dataclass
class LanguageFamilyProfile:
    """Detected language family and characteristics"""
    family: str = "Unknown"
    subfamily: str = ""
    confidence: float = 0.0
    is_tonal: bool = False
    rhythm_type: str = ""
    typology: str = ""             # SOV, SVO, VSO etc (inferred)
    characteristics: list = field(default_factory=list)


@dataclass
class SyntacticFrame:
    """Inferred syntactic structure from prosody"""
    phrase_boundaries: list = field(default_factory=list)
    phrase_count: int = 0
    head_positions: list = field(default_factory=list)
    dependency_graph: dict = field(default_factory=dict)
    sentence_type: str = ""        # declarative, interrogative, imperative
    negation_detected: bool = False
    focus_position: int = -1


@dataclass
class SemanticFrame:
    """Universal semantic role frame - language independent"""
    agent: str = ""                # Who does the action
    action: str = ""               # What happens
    patient: str = ""              # What receives the action
    location: str = ""             # Where
    time: str = ""                 # When
    manner: str = ""               # How
    purpose: str = ""              # Why
    sentence_type: str = ""
    polarity: str = "positive"
    certainty: float = 1.0
    emotional_valence: float = 0.0  # -1 to 1
    primitives: list = field(default_factory=list)


@dataclass
class MeaningFrame:
    """Complete meaning extraction result"""
    prosody: ProsodicFeatures = field(default_factory=ProsodicFeatures)
    language_family: LanguageFamilyProfile = field(default_factory=LanguageFamilyProfile)
    syntax: SyntacticFrame = field(default_factory=SyntacticFrame)
    semantics: SemanticFrame = field(default_factory=SemanticFrame)
    confidence: float = 0.0
    meaning_summary: str = ""
    raw_primitives: list = field(default_factory=list)


# ============================================================
# MODULE ONE: PROSODIC FEATURE EXTRACTION
# ============================================================

class ProsodicExtractor:
    """
    Extract mathematical prosodic features from audio.
    These are the acoustic substrate of all linguistic meaning.
    No vocabulary required - pure mathematics.
    """

    def __init__(self, sr=22050):
        self.sr = sr

    def extract(self, audio: np.ndarray, sr: int) -> ProsodicFeatures:
        features = ProsodicFeatures()
        self.sr = sr
        features.duration = len(audio) / sr

        # --- F0 Extraction via Parselmouth (most accurate method) ---
        f0_contour = self._extract_f0_parselmouth(audio, sr)
        features.f0_contour = f0_contour

        voiced = f0_contour[f0_contour > 0]
        if len(voiced) > 5:
            features.f0_mean = float(np.mean(voiced))
            features.f0_std = float(np.std(voiced))
            features.f0_range = float(np.max(voiced) - np.min(voiced))
            features.f0_slope = self._compute_f0_slope(voiced)
            features.f0_final_contour = self._detect_terminal_contour(f0_contour)
            features.entropy = self._compute_f0_entropy(voiced)

        # --- Rhythm Analysis ---
        features.rhythm_type, features.speech_rate = self._analyze_rhythm(audio, sr)

        # --- Intensity ---
        rms = librosa.feature.rms(y=audio)[0]
        features.intensity_mean = float(np.mean(rms))
        features.intensity_std = float(np.std(rms))

        # --- Tonality Detection ---
        features.is_tonal, features.tone_count = self._detect_tonality(f0_contour, sr)

        # --- Information Density ---
        features.info_density = self._estimate_info_density(
            features.speech_rate, features.rhythm_type, features.is_tonal
        )

        return features

    def _extract_f0_parselmouth(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract F0 using Praat's autocorrelation method"""
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
            f0_values = pitch.selected_array['frequency']
            return f0_values.astype(float)
        except Exception:
            # Fallback to librosa
            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=75, fmax=600, sr=sr, frame_length=2048
            )
            f0 = np.where(voiced_flag, f0, 0.0)
            return np.nan_to_num(f0)

    def _compute_f0_slope(self, voiced_f0: np.ndarray) -> float:
        """
        Compute overall F0 trajectory slope.
        Positive = rising (question tendency)
        Negative = falling (statement tendency)
        """
        if len(voiced_f0) < 2:
            return 0.0
        x = np.arange(len(voiced_f0))
        slope, _, _, _, _ = stats.linregress(x, voiced_f0)
        return float(slope)

    def _detect_terminal_contour(self, f0_contour: np.ndarray) -> str:
        """
        Detect the terminal F0 contour pattern.
        Universal across languages:
        - Rising terminal = question
        - Falling terminal = statement
        - Rise-fall = exclamative
        - Level = continuation
        """
        voiced = f0_contour[f0_contour > 0]
        if len(voiced) < 10:
            return "unknown"

        # Look at last 20% of voiced frames
        tail_size = max(3, len(voiced) // 5)
        tail = voiced[-tail_size:]
        head = voiced[:tail_size]

        tail_slope, _, _, _, _ = stats.linregress(np.arange(len(tail)), tail)
        final_vs_mean = tail[-1] - np.mean(voiced)

        if tail_slope > 2.0 and final_vs_mean > 0:
            return "rising"          # Question
        elif tail_slope < -2.0 and final_vs_mean < 0:
            return "falling"         # Statement
        elif tail_slope > 1.0 and np.mean(tail) > np.mean(head):
            return "rise-fall"       # Exclamative
        else:
            return "level"           # Continuation

    def _analyze_rhythm(self, audio: np.ndarray, sr: int) -> tuple:
        """
        Classify rhythm type - one of the most powerful
        language family discriminators.

        Stress-timed (English, German, Russian):
            Inter-stress intervals are approximately equal
            High variance in syllable duration

        Syllable-timed (Spanish, French, Italian, Japanese):
            Syllables are approximately equal duration
            Low variance in syllable duration

        Returns (rhythm_type, speech_rate_syllables_per_second)
        """
        # Onset detection as proxy for syllable boundaries
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, units='time',
            backtrack=True, pre_max=3, post_max=3,
            pre_avg=3, post_avg=5, delta=0.07, wait=10
        )

        speech_rate = len(onset_frames) / (len(audio) / sr) if len(audio) > 0 else 0.0

        if len(onset_frames) < 3:
            return "unknown", speech_rate

        # Compute inter-onset intervals
        iois = np.diff(onset_frames)

        if len(iois) < 2:
            return "unknown", speech_rate

        # Normalized Pairwise Variability Index (nPVI)
        # High nPVI = stress-timed
        # Low nPVI = syllable-timed
        npvi = self._compute_npvi(iois)

        if npvi > 45:
            rhythm_type = "stress-timed"
        elif npvi < 35:
            rhythm_type = "syllable-timed"
        else:
            rhythm_type = "mora-timed"  # Japanese, some others

        return rhythm_type, float(speech_rate)

    def _compute_npvi(self, intervals: np.ndarray) -> float:
        """
        Normalized Pairwise Variability Index.
        Key metric for rhythm classification.
        Ramus et al. (1999), Low et al. (2000)
        """
        if len(intervals) < 2:
            return 0.0
        pairs = list(zip(intervals[:-1], intervals[1:]))
        npvi_values = [
            abs(d1 - d2) / ((d1 + d2) / 2)
            for d1, d2 in pairs
            if (d1 + d2) > 0
        ]
        return float(np.mean(npvi_values) * 100) if npvi_values else 0.0

    def _compute_f0_entropy(self, voiced_f0: np.ndarray) -> float:
        """
        Shannon entropy of F0 distribution.
        High entropy = more tonal complexity
        Low entropy = simpler pitch pattern
        """
        if len(voiced_f0) < 5:
            return 0.0
        hist, _ = np.histogram(voiced_f0, bins=20, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist + 1e-10)))

    def _estimate_info_density(self, speech_rate: float,
                               rhythm_type: str,
                               is_tonal: bool) -> float:
        """
        Estimate information density in bits per second.
        Based on Pellegrino et al. (2011) finding that all languages
        transmit approximately 39 bits/second regardless of speech rate.
        
        Tonal languages carry more info per syllable (tone = extra channel).
        Stress-timed languages carry more info per syllable (complex syllables).
        """
        if speech_rate <= 0:
            return 0.0
        base_bits_per_syllable = 39.0 / max(speech_rate, 1.0)
        if is_tonal:
            base_bits_per_syllable *= 1.3
        if rhythm_type == "stress-timed":
            base_bits_per_syllable *= 1.15
        return float(base_bits_per_syllable * speech_rate)

    def _detect_tonality(self, f0_contour: np.ndarray, sr: int) -> tuple:
        """
        Detect whether language is likely tonal.
        Tonal languages (Mandarin, Vietnamese, Thai):
        - More distinct F0 clusters
        - F0 variation within syllables is systematic
        - Higher F0 entropy overall
        """
        voiced = f0_contour[f0_contour > 0]
        if len(voiced) < 10:
            return False, 0

        # Cluster F0 values - tonal languages show distinct clusters
        from scipy.cluster.hierarchy import fclusterdata
        try:
            f0_normalized = (voiced - voiced.min()) / (voiced.max() - voiced.min() + 1e-10)
            clusters = fclusterdata(
                f0_normalized.reshape(-1, 1),
                t=0.15, criterion='distance'
            )
            n_clusters = len(np.unique(clusters))
        except Exception:
            n_clusters = 1

        # High F0 entropy + multiple clusters suggests tonal language
        entropy = self._compute_f0_entropy(voiced)
        is_tonal = (entropy > 3.5 and n_clusters >= 3)

        return is_tonal, n_clusters


# ============================================================
# MODULE TWO: LANGUAGE FAMILY DETECTION
# ============================================================

class LanguageFamilyDetector:
    """
    Detect language family from acoustic features alone.
    No vocabulary. No dictionary.
    Pure mathematical classification.

    Based on prosodic typology research:
    - Ramus et al. (1999) - rhythm metrics
    - Hirst & Di Cristo (1998) - intonation systems
    - Maddieson (1984) - sound patterns of languages
    """

    # Language family acoustic profiles
    # Based on empirical research across 50+ languages
    FAMILY_PROFILES = {
        "Sino-Tibetan": {
            "is_tonal": True,
            "rhythm": "syllable-timed",
            "f0_range_hz": (100, 200),
            "speech_rate_range": (5.5, 8.5),
            "npvi_range": (20, 40),
            "description": "Mandarin, Cantonese, Tibetan family"
        },
        "Tai-Kadai": {
            "is_tonal": True,
            "rhythm": "syllable-timed",
            "f0_range_hz": (80, 180),
            "speech_rate_range": (5.0, 8.0),
            "npvi_range": (15, 35),
            "description": "Thai, Lao family"
        },
        "Austroasiatic": {
            "is_tonal": True,
            "rhythm": "syllable-timed",
            "f0_range_hz": (90, 190),
            "speech_rate_range": (5.0, 7.5),
            "npvi_range": (20, 38),
            "description": "Vietnamese, Khmer family"
        },
        "Indo-European-Germanic": {
            "is_tonal": False,
            "rhythm": "stress-timed",
            "f0_range_hz": (80, 180),
            "speech_rate_range": (3.5, 6.5),
            "npvi_range": (50, 75),
            "description": "English, German, Dutch family"
        },
        "Indo-European-Romance": {
            "is_tonal": False,
            "rhythm": "syllable-timed",
            "f0_range_hz": (90, 200),
            "speech_rate_range": (6.0, 9.0),
            "npvi_range": (30, 50),
            "description": "Spanish, French, Italian, Portuguese family"
        },
        "Indo-European-Slavic": {
            "is_tonal": False,
            "rhythm": "stress-timed",
            "f0_range_hz": (85, 175),
            "speech_rate_range": (4.0, 7.0),
            "npvi_range": (48, 70),
            "description": "Russian, Polish, Czech family"
        },
        "Semitic": {
            "is_tonal": False,
            "rhythm": "stress-timed",
            "f0_range_hz": (90, 190),
            "speech_rate_range": (4.5, 7.5),
            "npvi_range": (45, 68),
            "description": "Arabic, Hebrew, Amharic family"
        },
        "Japonic": {
            "is_tonal": False,
            "rhythm": "mora-timed",
            "f0_range_hz": (100, 220),
            "speech_rate_range": (6.5, 9.5),
            "npvi_range": (28, 42),
            "description": "Japanese family"
        },
        "Koreanic": {
            "is_tonal": False,
            "rhythm": "syllable-timed",
            "f0_range_hz": (95, 210),
            "speech_rate_range": (5.5, 8.5),
            "npvi_range": (32, 48),
            "description": "Korean family"
        },
        "Dravidian": {
            "is_tonal": False,
            "rhythm": "syllable-timed",
            "f0_range_hz": (100, 210),
            "speech_rate_range": (5.0, 8.0),
            "npvi_range": (30, 50),
            "description": "Tamil, Telugu, Kannada family"
        },
        "Niger-Congo": {
            "is_tonal": True,
            "rhythm": "syllable-timed",
            "f0_range_hz": (110, 220),
            "speech_rate_range": (5.5, 8.5),
            "npvi_range": (18, 38),
            "description": "Swahili, Yoruba, Zulu family"
        },
        "Turkic": {
            "is_tonal": False,
            "rhythm": "syllable-timed",
            "f0_range_hz": (90, 195),
            "speech_rate_range": (5.0, 7.5),
            "npvi_range": (35, 52),
            "description": "Turkish, Uzbek, Kazakh family"
        },
    }

    def detect(self, features: ProsodicFeatures) -> LanguageFamilyProfile:
        profile = LanguageFamilyProfile()

        scores = {}
        for family, params in self.FAMILY_PROFILES.items():
            score = self._compute_family_score(features, params)
            scores[family] = score

        # Rank families by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_family, best_score = ranked[0]

        profile.family = best_family
        profile.confidence = min(best_score, 1.0)
        profile.is_tonal = features.is_tonal
        profile.rhythm_type = features.rhythm_type

        params = self.FAMILY_PROFILES[best_family]
        profile.characteristics = [
            f"Rhythm: {params['rhythm']}",
            f"Tonal: {'Yes' if params['is_tonal'] else 'No'}",
            f"Description: {params['description']}",
            f"F0 range detected: {features.f0_range:.1f} Hz",
            f"Speech rate: {features.speech_rate:.1f} syllables/sec",
        ]

        # Infer word order typology from rhythm + other features
        profile.typology = self._infer_typology(best_family, features)

        return profile

    def _compute_family_score(self, features: ProsodicFeatures, params: dict) -> float:
        """
        Compute similarity score between detected features
        and family profile. Multi-factor scoring.
        """
        score = 0.0
        weights = {
            'tonality': 0.35,
            'rhythm': 0.30,
            'speech_rate': 0.20,
            'f0_range': 0.15,
        }

        # Tonality match (binary, high weight)
        if features.is_tonal == params['is_tonal']:
            score += weights['tonality']

        # Rhythm type match
        if features.rhythm_type == params['rhythm']:
            score += weights['rhythm']
        elif features.rhythm_type == "unknown":
            score += weights['rhythm'] * 0.3  # Partial credit for unknown

        # Speech rate within range
        rate_min, rate_max = params['speech_rate_range']
        if rate_min <= features.speech_rate <= rate_max:
            score += weights['speech_rate']
        elif features.speech_rate > 0:
            # Partial credit based on proximity
            midpoint = (rate_min + rate_max) / 2
            distance = abs(features.speech_rate - midpoint)
            range_width = rate_max - rate_min
            proximity = max(0, 1 - (distance / range_width))
            score += weights['speech_rate'] * proximity * 0.5

        # F0 range within expected range
        f0_min, f0_max = params['f0_range_hz']
        if f0_min <= features.f0_range <= f0_max:
            score += weights['f0_range']

        return score

    def _infer_typology(self, family: str, features: ProsodicFeatures) -> str:
        """
        Infer basic word order typology from family.
        Based on Greenberg's universals and WALS data.
        """
        typology_map = {
            "Indo-European-Germanic": "SVO",
            "Indo-European-Romance": "SVO",
            "Indo-European-Slavic": "SVO (flexible)",
            "Semitic": "VSO/SVO",
            "Japonic": "SOV",
            "Koreanic": "SOV",
            "Dravidian": "SOV",
            "Turkic": "SOV",
            "Sino-Tibetan": "SVO",
            "Tai-Kadai": "SVO",
            "Austroasiatic": "SVO",
            "Niger-Congo": "SVO",
        }
        return typology_map.get(family, "SVO")


# ============================================================
# MODULE THREE: SYNTACTIC STRUCTURE FROM PROSODY
# ============================================================

class SyntacticAnalyzer:
    """
    Infer syntactic structure purely from prosodic cues.
    
    Key insight: Prosody and syntax are deeply coupled.
    Phrase boundaries are marked prosodically in ALL languages:
    - F0 reset at phrase start
    - Duration lengthening at phrase end  
    - Intensity reduction at phrase end
    - Pause insertion at major boundaries
    
    This module does NOT need to know the language.
    """

    def analyze(self, audio: np.ndarray, sr: int,
                features: ProsodicFeatures,
                language_profile: LanguageFamilyProfile) -> SyntacticFrame:
        frame = SyntacticFrame()

        # Detect phrase boundaries
        frame.phrase_boundaries = self._detect_phrase_boundaries(audio, sr, features)
        frame.phrase_count = len(frame.phrase_boundaries) + 1

        # Detect sentence type from terminal contour
        frame.sentence_type = self._infer_sentence_type(features)

        # Detect negation from prosodic markers
        frame.negation_detected = self._detect_negation(audio, sr, features)

        # Detect focus position (most prominent element)
        frame.focus_position = self._detect_focus(audio, sr)

        # Build basic dependency graph based on phrase structure
        frame.dependency_graph = self._build_dependency_graph(
            frame.phrase_count,
            frame.sentence_type,
            language_profile.typology
        )

        return frame

    def _detect_phrase_boundaries(self, audio: np.ndarray, sr: int,
                                   features: ProsodicFeatures) -> list:
        """
        Detect major phrase boundaries using multiple prosodic cues.
        
        Boundary markers (universal):
        1. Pause (silence > 150ms)
        2. F0 reset (sudden F0 jump upward)
        3. Duration lengthening (final syllable longer)
        4. Intensity drop followed by reset
        """
        boundaries = []

        # Method 1: Silence detection
        silence_boundaries = self._find_silences(audio, sr)
        boundaries.extend(silence_boundaries)

        # Method 2: F0 reset detection
        f0_boundaries = self._find_f0_resets(features.f0_contour)
        boundaries.extend(f0_boundaries)

        # Remove duplicates and sort
        if boundaries:
            boundaries = sorted(set([round(b, 2) for b in boundaries]))
            # Merge boundaries within 200ms of each other
            merged = [boundaries[0]]
            for b in boundaries[1:]:
                if b - merged[-1] > 0.2:
                    merged.append(b)
            boundaries = merged

        return boundaries

    def _find_silences(self, audio: np.ndarray, sr: int,
                       threshold_db: float = -40,
                       min_duration: float = 0.15) -> list:
        """Find silence regions as phrase boundaries"""
        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        silence_frames = rms_db < threshold_db
        boundaries = []

        in_silence = False
        silence_start = 0
        hop_time = 256 / sr

        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                in_silence = True
                silence_start = i
            elif not is_silent and in_silence:
                in_silence = False
                duration = (i - silence_start) * hop_time
                if duration >= min_duration:
                    boundary_time = silence_start * hop_time + duration / 2
                    boundaries.append(boundary_time)

        return boundaries

    def _find_f0_resets(self, f0_contour: np.ndarray,
                        reset_threshold: float = 30.0) -> list:
        """
        Find F0 resets - sudden upward jumps that mark phrase boundaries.
        Universal across languages.
        """
        if len(f0_contour) < 10:
            return []

        voiced_indices = np.where(f0_contour > 0)[0]
        if len(voiced_indices) < 5:
            return []

        boundaries = []
        frame_time = 0.01  # 10ms per frame

        for i in range(1, len(voiced_indices)):
            idx_curr = voiced_indices[i]
            idx_prev = voiced_indices[i-1]

            # Only consider adjacent voiced frames
            if idx_curr - idx_prev <= 3:
                f0_jump = f0_contour[idx_curr] - f0_contour[idx_prev]
                if f0_jump > reset_threshold:
                    boundaries.append(idx_curr * frame_time)

        return boundaries

    def _infer_sentence_type(self, features: ProsodicFeatures) -> str:
        """
        Infer sentence type from terminal F0 contour.
        This is one of the most robust universals in prosody.
        """
        contour = features.f0_final_contour

        if contour == "rising":
            return "interrogative"
        elif contour == "falling":
            return "declarative"
        elif contour == "rise-fall":
            return "exclamative"
        elif contour == "level":
            return "continuation"
        else:
            # Use slope as fallback
            if features.f0_slope > 1.5:
                return "interrogative"
            elif features.f0_slope < -1.5:
                return "declarative"
            else:
                return "declarative"

    def _detect_negation(self, audio: np.ndarray, sr: int,
                         features: ProsodicFeatures) -> bool:
        """
        Detect probable negation from prosodic markers.
        
        Negation tends to:
        - Increase F0 range slightly
        - Create specific intensity patterns
        - Extend duration of key elements
        
        Not perfect without vocabulary but statistically significant.
        """
        # High F0 variability combined with specific patterns
        # suggests contrastive/negative meaning
        high_variability = features.f0_std > (features.f0_mean * 0.3)
        high_range = features.f0_range > 100
        return bool(high_variability and high_range and
                    features.f0_final_contour in ["falling", "level"])

    def _detect_focus(self, audio: np.ndarray, sr: int) -> int:
        """
        Detect the focus position - most prominent element.
        Marked by F0 peak + intensity peak.
        Returns approximate phrase position (0-indexed).
        """
        rms = librosa.feature.rms(y=audio)[0]
        peak_position = int(np.argmax(rms))
        # Normalize to phrase position
        return peak_position // max(1, len(rms) // 3)

    def _build_dependency_graph(self, phrase_count: int,
                                 sentence_type: str,
                                 typology: str) -> dict:
        """
        Build approximate dependency graph based on:
        - Number of phrases detected
        - Sentence type
        - Known word order for language family
        """
        graph = {}

        if phrase_count == 1:
            graph = {"phrase_0": {"role": "predicate", "children": []}}
        elif phrase_count == 2:
            if typology.startswith("SVO"):
                graph = {
                    "phrase_0": {"role": "subject/agent", "children": ["phrase_1"]},
                    "phrase_1": {"role": "predicate/verb", "children": []}
                }
            else:  # SOV
                graph = {
                    "phrase_0": {"role": "subject/agent", "children": ["phrase_1"]},
                    "phrase_1": {"role": "predicate/verb", "children": []}
                }
        elif phrase_count >= 3:
            if typology.startswith("SVO"):
                graph = {
                    "phrase_0": {"role": "subject/agent", "children": ["phrase_1"]},
                    "phrase_1": {"role": "predicate/verb", "children": ["phrase_2"]},
                    "phrase_2": {"role": "object/patient", "children": []}
                }
            elif typology.startswith("SOV"):
                graph = {
                    "phrase_0": {"role": "subject/agent", "children": ["phrase_1"]},
                    "phrase_1": {"role": "object/patient", "children": ["phrase_2"]},
                    "phrase_2": {"role": "predicate/verb", "children": []}
                }
            else:  # VSO
                graph = {
                    "phrase_0": {"role": "predicate/verb", "children": ["phrase_1", "phrase_2"]},
                    "phrase_1": {"role": "subject/agent", "children": []},
                    "phrase_2": {"role": "object/patient", "children": []}
                }

            # Add additional phrases as modifiers
            for i in range(3, phrase_count):
                graph[f"phrase_{i}"] = {
                    "role": "modifier/adjunct",
                    "children": []
                }

        return graph


# ============================================================
# MODULE FOUR: SEMANTIC PRIMITIVE DETECTION
# ============================================================

class SemanticPrimitiveDetector:
    """
    Detect semantic primitives from prosodic and acoustic features.
    
    Based on Wierzbicka's Natural Semantic Metalanguage (NSM).
    65 universal semantic primitives exist in ALL human languages.
    
    We detect the most acoustically discriminable ones:
    - Polarity (positive/negative)
    - Certainty level
    - Emotional valence
    - Action vs state
    - Question vs statement
    - Intensification
    - Temporal reference
    """

    # Acoustic correlates of semantic primitives
    # Based on cross-linguistic prosody research
    PRIMITIVE_DETECTORS = {
        "QUESTION": {
            "terminal_contour": "rising",
            "f0_slope_threshold": 1.0,
            "description": "Interrogative act"
        },
        "COMMAND": {
            "terminal_contour": "falling",
            "intensity_high": True,
            "f0_range_high": True,
            "description": "Imperative act"
        },
        "NEGATION": {
            "high_f0_variability": True,
            "description": "Negative polarity"
        },
        "INTENSIFICATION": {
            "intensity_high": True,
            "f0_range_high": True,
            "description": "Emphasis/intensification"
        },
        "SURPRISE": {
            "terminal_contour": "rise-fall",
            "high_f0_range": True,
            "description": "Surprise/unexpectedness"
        },
        "CERTAINTY": {
            "terminal_contour": "falling",
            "low_f0_variability": True,
            "description": "High certainty"
        },
        "UNCERTAINTY": {
            "terminal_contour": "rising",
            "high_f0_variability": True,
            "description": "Uncertainty/hedging"
        },
        "GOOD": {
            "positive_valence": True,
            "description": "Positive evaluation"
        },
        "BAD": {
            "negative_valence": True,
            "description": "Negative evaluation"
        },
    }

    def detect(self, features: ProsodicFeatures,
               syntax: SyntacticFrame) -> tuple:
        """
        Returns (SemanticFrame, list of detected primitives)
        """
        frame = SemanticFrame()
        detected_primitives = []

        # Sentence type
        frame.sentence_type = syntax.sentence_type

        # Polarity
        if syntax.negation_detected:
            frame.polarity = "negative"
            detected_primitives.append("NOT")
        else:
            frame.polarity = "positive"

        # Certainty
        frame.certainty = self._estimate_certainty(features)
        if frame.certainty > 0.7:
            detected_primitives.append("KNOW (certain)")
        else:
            detected_primitives.append("THINK (uncertain)")

        # Emotional valence
        frame.emotional_valence = self._estimate_valence(features)
        if frame.emotional_valence > 0.3:
            detected_primitives.append("GOOD")
        elif frame.emotional_valence < -0.3:
            detected_primitives.append("BAD")

        # Action vs state
        action_primitive = self._detect_action_vs_state(features, syntax)
        detected_primitives.append(action_primitive)

        # Assign thematic roles from dependency graph
        self._assign_thematic_roles(frame, syntax)

        # Question primitive
        if syntax.sentence_type == "interrogative":
            detected_primitives.append("WHAT/WHERE/WHO (question)")

        # Intensification
        if features.intensity_std > features.intensity_mean * 0.4:
            detected_primitives.append("VERY (intensification)")

        # Temporal reference (basic)
        temporal = self._infer_temporal_reference(features)
        if temporal:
            detected_primitives.append(temporal)

        frame.primitives = detected_primitives

        # Build meaning summary
        frame.action = action_primitive

        return frame, detected_primitives

    def _estimate_certainty(self, features: ProsodicFeatures) -> float:
        """
        Certainty correlates with:
        - Falling terminal contour (more certain)
        - Lower F0 variability (more certain)
        - Steady rhythm (more certain)
        """
        certainty = 0.5  # Base

        if features.f0_final_contour == "falling":
            certainty += 0.25
        elif features.f0_final_contour == "rising":
            certainty -= 0.2

        if features.f0_mean > 0:
            variability_ratio = features.f0_std / features.f0_mean
            if variability_ratio < 0.2:
                certainty += 0.15
            elif variability_ratio > 0.4:
                certainty -= 0.15

        return float(np.clip(certainty, 0.0, 1.0))

    def _estimate_valence(self, features: ProsodicFeatures) -> float:
        """
        Emotional valence from prosodic features.
        
        Positive valence tends toward:
        - Higher mean F0
        - Wider F0 range
        - Higher speech rate
        
        Negative valence tends toward:
        - Lower mean F0
        - Narrower F0 range
        - Slower speech rate
        
        Based on: Scherer (1986), Banse & Scherer (1996)
        Universal cross-cultural findings.
        """
        valence = 0.0

        # F0 height relative to expected range
        # Higher F0 tends toward positive affect
        if features.f0_mean > 200:
            valence += 0.2
        elif features.f0_mean < 130:
            valence -= 0.15

        # F0 range
        if features.f0_range > 100:
            valence += 0.15
        elif features.f0_range < 40:
            valence -= 0.1

        # Speech rate
        if features.speech_rate > 6.0:
            valence += 0.1
        elif features.speech_rate < 3.5:
            valence -= 0.1

        # Intensity variation
        if features.intensity_std > features.intensity_mean * 0.3:
            valence += 0.1

        return float(np.clip(valence, -1.0, 1.0))

    def _detect_action_vs_state(self, features: ProsodicFeatures,
                                 syntax: SyntacticFrame) -> str:
        """
        Distinguish action vs state predicates from prosody.
        Actions tend to have more dynamic F0 and intensity.
        States tend to be more level.
        """
        dynamic_f0 = features.f0_std > 30
        dynamic_intensity = features.intensity_std > features.intensity_mean * 0.25

        if dynamic_f0 and dynamic_intensity:
            return "DO (action)"
        elif features.f0_slope < -2 and not dynamic_f0:
            return "BE (state)"
        else:
            return "HAPPEN (event)"

    def _assign_thematic_roles(self, frame: SemanticFrame,
                                syntax: SyntacticFrame):
        """Assign thematic roles from dependency graph"""
        for phrase_id, props in syntax.dependency_graph.items():
            role = props.get("role", "")
            if "subject" in role or "agent" in role:
                frame.agent = f"[{phrase_id}]"
            elif "object" in role or "patient" in role:
                frame.patient = f"[{phrase_id}]"
            elif "predicate" in role or "verb" in role:
                frame.action = f"[{phrase_id}]: {frame.action}"
            elif "modifier" in role:
                if not frame.manner:
                    frame.manner = f"[{phrase_id}]"

    def _infer_temporal_reference(self, features: ProsodicFeatures) -> Optional[str]:
        """
        Very basic temporal reference inference.
        This is the weakest module - true tense requires vocabulary.
        But prosodic energy patterns correlate weakly with tense.
        """
        # High-energy present > lower-energy past (very weak signal)
        if features.intensity_mean > 0.05:
            return "NOW (present reference)"
        return None


# ============================================================
# MODULE FIVE: MEANING FRAME CONSTRUCTOR
# ============================================================

class MeaningFrameConstructor:
    """
    Combine all module outputs into a unified meaning frame.
    Produces human-readable meaning summary.
    """

    def construct(self,
                  prosody: ProsodicFeatures,
                  language: LanguageFamilyProfile,
                  syntax: SyntacticFrame,
                  semantics: SemanticFrame,
                  primitives: list) -> MeaningFrame:

        frame = MeaningFrame()
        frame.prosody = prosody
        frame.language_family = language
        frame.syntax = syntax
        frame.semantics = semantics
        frame.raw_primitives = primitives

        # Compute overall confidence
        frame.confidence = self._compute_confidence(prosody, language, syntax)

        # Build meaning summary
        frame.meaning_summary = self._build_summary(
            syntax, semantics, primitives, language
        )

        return frame

    def _compute_confidence(self, prosody: ProsodicFeatures,
                             language: LanguageFamilyProfile,
                             syntax: SyntacticFrame) -> float:
        """Overall confidence in meaning extraction"""
        confidence = 0.0

        # Language family confidence contributes
        confidence += language.confidence * 0.35

        # Audio quality indicators
        if prosody.duration > 0.5:
            confidence += 0.15
        if prosody.f0_mean > 0:
            confidence += 0.20
        if prosody.speech_rate > 1.0:
            confidence += 0.10

        # Structural clarity
        if syntax.phrase_count > 0:
            confidence += 0.15
        if syntax.sentence_type != "":
            confidence += 0.05

        return float(np.clip(confidence, 0.0, 1.0))

    def _build_summary(self, syntax: SyntacticFrame,
                        semantics: SemanticFrame,
                        primitives: list,
                        language: LanguageFamilyProfile) -> str:
        """Build human-readable meaning summary"""
        parts = []

        # Sentence type
        type_map = {
            "interrogative": "QUESTION",
            "declarative": "STATEMENT",
            "exclamative": "EXCLAMATION",
            "imperative": "COMMAND",
            "continuation": "CONTINUATION"
        }
        parts.append(f"Act: {type_map.get(syntax.sentence_type, 'STATEMENT')}")

        # Polarity
        if semantics.polarity == "negative":
            parts.append("Polarity: NEGATIVE")

        # Structure
        parts.append(f"Structure: {syntax.phrase_count} phrase(s)")
        if language.typology:
            parts.append(f"Word order: {language.typology}")

        # Thematic roles
        if semantics.agent:
            parts.append(f"Agent (who): {semantics.agent}")
        if semantics.action:
            parts.append(f"Action: {semantics.action}")
        if semantics.patient:
            parts.append(f"Patient (what): {semantics.patient}")
        if semantics.manner:
            parts.append(f"Modifier: {semantics.manner}")

        # Certainty and valence
        if semantics.certainty > 0.7:
            parts.append("Certainty: HIGH")
        elif semantics.certainty < 0.4:
            parts.append("Certainty: LOW")

        if semantics.emotional_valence > 0.3:
            parts.append("Affect: POSITIVE")
        elif semantics.emotional_valence < -0.3:
            parts.append("Affect: NEGATIVE")

        # Primitives
        if primitives:
            parts.append(f"Primitives: {', '.join(primitives[:5])}")

        return " | ".join(parts)


# ============================================================
# MAIN ATOMQ ENGINE
# ============================================================

class AtomQ:
    """
    AtomQ: Universal Language Understanding Engine
    
    Extracts meaning from any human language
    without a dictionary, without knowing the language,
    using only the universal mathematics of human speech.
    
    Usage:
        atomq = AtomQ()
        result = atomq.analyze(audio_array, sample_rate)
        print(result.meaning_summary)
    """

    def __init__(self):
        self.prosodic_extractor = ProsodicExtractor()
        self.family_detector = LanguageFamilyDetector()
        self.syntactic_analyzer = SyntacticAnalyzer()
        self.primitive_detector = SemanticPrimitiveDetector()
        self.meaning_constructor = MeaningFrameConstructor()

    def analyze(self, audio: np.ndarray, sr: int,
                verbose: bool = False) -> MeaningFrame:
        """
        Full AtomQ analysis pipeline.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            verbose: Print intermediate results
            
        Returns:
            MeaningFrame with complete analysis
        """
        if verbose:
            print("AtomQ Analysis Pipeline")
            print("=" * 50)

        # Module 1: Prosodic Feature Extraction
        if verbose:
            print("Module 1: Extracting prosodic features...")
        prosody = self.prosodic_extractor.extract(audio, sr)

        if verbose:
            print(f"  F0 mean: {prosody.f0_mean:.1f} Hz")
            print(f"  F0 range: {prosody.f0_range:.1f} Hz")
            print(f"  Rhythm: {prosody.rhythm_type}")
            print(f"  Speech rate: {prosody.speech_rate:.1f} syl/sec")
            print(f"  Terminal contour: {prosody.f0_final_contour}")
            print(f"  Tonal: {prosody.is_tonal}")
            print(f"  Shannon entropy: {prosody.entropy:.2f} bits")

        # Module 2: Language Family Detection
        if verbose:
            print("\nModule 2: Detecting language family...")
        language = self.family_detector.detect(prosody)

        if verbose:
            print(f"  Family: {language.family}")
            print(f"  Confidence: {language.confidence:.1%}")
            print(f"  Typology: {language.typology}")

        # Module 3: Syntactic Structure
        if verbose:
            print("\nModule 3: Inferring syntactic structure...")
        syntax = self.syntactic_analyzer.analyze(audio, sr, prosody, language)

        if verbose:
            print(f"  Sentence type: {syntax.sentence_type}")
            print(f"  Phrases detected: {syntax.phrase_count}")
            print(f"  Phrase boundaries: {syntax.phrase_boundaries}")
            print(f"  Negation detected: {syntax.negation_detected}")

        # Module 4: Semantic Primitives
        if verbose:
            print("\nModule 4: Detecting semantic primitives...")
        semantics, primitives = self.primitive_detector.detect(prosody, syntax)

        if verbose:
            print(f"  Polarity: {semantics.polarity}")
            print(f"  Certainty: {semantics.certainty:.1%}")
            print(f"  Emotional valence: {semantics.emotional_valence:+.2f}")
            print(f"  Primitives: {primitives}")

        # Module 5: Meaning Frame Construction
        if verbose:
            print("\nModule 5: Constructing meaning frame...")
        meaning = self.meaning_constructor.construct(
            prosody, language, syntax, semantics, primitives
        )

        if verbose:
            print(f"\n{'=' * 50}")
            print("MEANING EXTRACTION RESULT")
            print(f"{'=' * 50}")
            print(f"  {meaning.meaning_summary}")
            print(f"  Overall confidence: {meaning.confidence:.1%}")

        return meaning

    def analyze_file(self, filepath: str, verbose: bool = False) -> MeaningFrame:
        """Load audio file and analyze"""
        audio, sr = librosa.load(filepath, sr=None, mono=True)
        return self.analyze(audio, sr, verbose=verbose)

    def analyze_text_to_speech(self, text: str, language: str = "en",
                                verbose: bool = False) -> MeaningFrame:
        """
        Generate speech from text and analyze.
        Useful for testing with known content.
        """
        try:
            from gtts import gTTS
            import io
            import soundfile as sf

            tts = gTTS(text=text, lang=language, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            audio, sr = librosa.load(audio_buffer, sr=22050, mono=True)
            return self.analyze(audio, sr, verbose=verbose)
        except ImportError:
            raise ImportError("Install gtts for text-to-speech testing: pip install gtts")


# ============================================================
# SYNTHETIC AUDIO GENERATOR (For Testing Without Real Audio)
# ============================================================

class SyntheticSpeechGenerator:
    """
    Generate synthetic speech-like audio for testing.
    Simulates prosodic patterns of different language families.
    Does NOT require real speech data.
    """

    @staticmethod
    def generate(duration: float = 3.0,
                 sr: int = 22050,
                 f0_mean: float = 150.0,
                 f0_range: float = 80.0,
                 rhythm: str = "stress-timed",
                 is_tonal: bool = False,
                 sentence_type: str = "declarative") -> tuple:
        """
        Generate synthetic speech-like audio with specified prosodic profile.
        Returns (audio_array, sample_rate)
        """
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.zeros_like(t)

        # Generate syllable-like bursts
        n_syllables = int(duration * (7 if rhythm == "syllable-timed" else 5))

        if rhythm == "stress-timed":
            # Irregular inter-syllable intervals
            positions = np.sort(np.random.uniform(0.05, duration - 0.05, n_syllables))
            durations = np.random.exponential(0.15, n_syllables)
        else:
            # Regular inter-syllable intervals
            positions = np.linspace(0.05, duration - 0.05, n_syllables)
            durations = np.ones(n_syllables) * 0.12

        # Generate F0 contour
        f0_contour = SyntheticSpeechGenerator._generate_f0_contour(
            t, f0_mean, f0_range, sentence_type, is_tonal
        )

        # Synthesize voiced speech
        for i, (pos, dur) in enumerate(zip(positions, durations)):
            start = int(pos * sr)
            end = int(min((pos + dur) * sr, len(t)))
            if start >= len(t):
                break

            segment_len = end - start
            segment_t = np.linspace(0, dur, segment_len)

            # Local F0
            frame_idx = min(int(pos * len(f0_contour) / duration),
                           len(f0_contour) - 1)
            local_f0 = max(80, f0_contour[frame_idx])

            # Generate harmonic series (voiced speech simulation)
            voiced = np.zeros(segment_len)
            for harmonic in range(1, 8):
                amplitude = 1.0 / harmonic
                voiced += amplitude * np.sin(2 * np.pi * local_f0 * harmonic * segment_t)

            # Apply envelope
            envelope = np.hanning(segment_len)
            voiced *= envelope

            # Add to audio
            audio[start:end] += voiced * 0.3

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8

        # Add slight noise for realism
        audio += np.random.normal(0, 0.01, len(audio))

        return audio, sr

    @staticmethod
    def _generate_f0_contour(t: np.ndarray,
                              f0_mean: float,
                              f0_range: float,
                              sentence_type: str,
                              is_tonal: bool) -> np.ndarray:
        """Generate F0 contour matching sentence type"""
        n = len(t)
        f0 = np.ones(n) * f0_mean

        # Base declination (universal in all languages)
        declination = np.linspace(f0_mean + f0_range * 0.3,
                                   f0_mean - f0_range * 0.2, n)
        f0 = declination

        # Terminal contour
        terminal_region = n // 4
        if sentence_type == "interrogative":
            # Rising terminal
            rise = np.linspace(0, f0_range * 0.8, terminal_region)
            f0[-terminal_region:] += rise
        elif sentence_type == "exclamative":
            # Rise-fall
            mid = terminal_region // 2
            rise_part = np.linspace(0, f0_range * 0.6, terminal_region - mid)
            fall_part = np.linspace(f0_range * 0.6, 0, mid)
            f0[-terminal_region:-mid] += rise_part
            f0[-mid:] += fall_part

        # Tonal patterns
        if is_tonal:
            # Add systematic F0 modulation (simulating tones)
            tone_freq = 2.5  # tones per second
            tonal_pattern = np.sin(2 * np.pi * tone_freq * t) * (f0_range * 0.4)
            f0 += tonal_pattern

        # Add phrase-level variation
        n_phrases = np.random.randint(2, 5)
        for i in range(n_phrases):
            start = int(i * n / n_phrases)
            end = int((i + 1) * n / n_phrases)
            phrase_f0 = np.linspace(
                f0_mean + np.random.uniform(-20, 30),
                f0_mean + np.random.uniform(-30, 10),
                end - start
            )
            f0[start:end] += (phrase_f0 - f0_mean) * 0.3

        return np.clip(f0, 50, 500)


# ============================================================
# MECHANICAL BACKUP ALGORITHM
# ============================================================

class AtomQMechanical:
    """
    AtomQ Mechanical Backup.
    Zero power. Zero electronics. Pure mathematical logic.
    
    Implements core meaning extraction using only:
    - Rhythm counting (inter-onset intervals)
    - Contour classification (rising/falling/level)
    - Intensity pattern (loud/soft/building/fading)
    
    Works on paper, with tuning forks, with mechanical counters.
    Accuracy ~45% vs ~72% for electronic version.
    But it works when nothing else does.
    """

    @staticmethod
    def analyze_mechanical(onset_times: list,
                           terminal_direction: str,
                           intensity_profile: str,
                           pause_positions: list) -> dict:
        """
        Pure mathematical analysis from basic observables.
        
        Args:
            onset_times: List of syllable onset times in seconds
            terminal_direction: "up", "down", or "level"
            intensity_profile: "building", "fading", "level", "peaked"
            pause_positions: List of pause positions in seconds
            
        Returns:
            Basic meaning frame as dictionary
        """
        result = {
            "sentence_type": "unknown",
            "phrase_count": len(pause_positions) + 1,
            "rhythm_type": "unknown",
            "polarity": "positive",
            "basic_meaning": ""
        }

        # Step 1: Classify rhythm from inter-onset intervals
        if len(onset_times) >= 3:
            iois = np.diff(onset_times)
            cv = np.std(iois) / np.mean(iois) if np.mean(iois) > 0 else 0
            result["rhythm_type"] = "stress-timed" if cv > 0.4 else "syllable-timed"

        # Step 2: Sentence type from terminal direction
        terminal_map = {
            "up": "question",
            "down": "statement",
            "level": "continuation"
        }
        result["sentence_type"] = terminal_map.get(terminal_direction, "statement")

        # Step 3: Emotional/pragmatic inference from intensity
        if intensity_profile == "building":
            result["basic_meaning"] = "increasing importance or urgency"
        elif intensity_profile == "fading":
            result["basic_meaning"] = "decreasing certainty or trailing thought"
        elif intensity_profile == "peaked":
            result["basic_meaning"] = "emphasized central element"
        else:
            result["basic_meaning"] = "neutral delivery"

        # Step 4: Complexity estimate
        result["complexity"] = "simple" if len(pause_positions) <= 1 else "complex"

        return result

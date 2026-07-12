import librosa
import numpy as np
import scipy.signal
from scipy.spatial.distance import cdist

NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Key mappings (Name, Uses_Flats)
MAJOR_KEYS = [
    ("C Major", False), ("Db Major", True), ("D Major", False), ("Eb Major", True),
    ("E Major", False), ("F Major", True), ("F# Major", False), ("G Major", False),
    ("Ab Major", True), ("A Major", False), ("Bb Major", True), ("B Major", False)
]

MINOR_KEYS = [
    ("C Minor", True), ("C# Minor", False), ("D Minor", True), ("Eb Minor", True),
    ("E Minor", False), ("F Minor", True), ("F# Minor", False), ("G Minor", True),
    ("G# Minor", False), ("A Minor", False), ("Bb Minor", True), ("B Minor", False)
]

def generate_templates(style='Pop', transpose_semitones=0, use_flats=False):
    templates = []
    labels = []
    
    notes_array = NOTES_FLAT if use_flats else NOTES_SHARP
    
    if style == 'Pop' or style == 'Standard':
        # 24 Chords: Major and Minor
        for root in range(12):
            root_transposed = (root + transpose_semitones) % 12
            # Major
            template = np.zeros(12)
            template[[root, (root+4)%12, (root+7)%12]] = 1
            templates.append(template)
            labels.append(f"{notes_array[root_transposed]}maj")
            
            # Minor
            template = np.zeros(12)
            template[[root, (root+3)%12, (root+7)%12]] = 1
            templates.append(template)
            labels.append(f"{notes_array[root_transposed]}m")
            
    elif style == 'Jazz':
        # 48 Chords: Maj7, Min7, Dom7, m7b5
        for root in range(12):
            root_transposed = (root + transpose_semitones) % 12
            # Maj7
            template = np.zeros(12)
            template[[root, (root+4)%12, (root+7)%12, (root+11)%12]] = 1
            templates.append(template)
            labels.append(f"{notes_array[root_transposed]}maj7")
            
            # Min7
            template = np.zeros(12)
            template[[root, (root+3)%12, (root+7)%12, (root+10)%12]] = 1
            templates.append(template)
            labels.append(f"{notes_array[root_transposed]}m7")
            
            # Dom7
            template = np.zeros(12)
            template[[root, (root+4)%12, (root+7)%12, (root+10)%12]] = 1
            templates.append(template)
            labels.append(f"{notes_array[root_transposed]}7")
            
            # m7b5
            template = np.zeros(12)
            template[[root, (root+3)%12, (root+6)%12, (root+10)%12]] = 1
            templates.append(template)
            labels.append(f"{notes_array[root_transposed]}m7b5")
    
    return np.array(templates), labels

def estimate_key(chromagram):
    chroma_sum = np.sum(chromagram, axis=1)
    
    # Krumhansl-Schmuckler profiles
    maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    maj_profile = maj_profile / np.linalg.norm(maj_profile)
    min_profile = min_profile / np.linalg.norm(min_profile)
    
    if np.linalg.norm(chroma_sum) > 0:
        chroma_sum = chroma_sum / np.linalg.norm(chroma_sum)
        
    best_corr = -1
    best_key_str = "Unknown"
    best_root = 0
    best_is_major = True
    
    for i in range(12):
        maj_p = np.roll(maj_profile, i)
        min_p = np.roll(min_profile, i)
        
        corr_maj = np.correlate(chroma_sum, maj_p)[0]
        corr_min = np.correlate(chroma_sum, min_p)[0]
        
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key_str = MAJOR_KEYS[i][0]
            best_root = i
            best_is_major = True
            
        if corr_min > best_corr:
            best_corr = corr_min
            best_key_str = MINOR_KEYS[i][0]
            best_root = i
            best_is_major = False
            
    return best_key_str, best_root, best_is_major

def analyze_audio(audio_path, style='Pop', smooth_kernel=15):
    """
    Extracts chords from an audio file.
    smooth_kernel: Number of frames to smooth over (e.g. 15 frames ~= 1-2 seconds)
    """
    # 1. Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 2. Extract harmonic component
    y_harmonic, _ = librosa.effects.hpss(y)
    
    # 3. Compute Chromagram
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    # Estimate Key
    estimated_key_str, estimated_root, estimated_is_major = estimate_key(chromagram)
    
    # Normalize chromagram frames
    chroma_norms = np.linalg.norm(chromagram, axis=0)
    chroma_norms[chroma_norms == 0] = 1 # prevent div by zero
    chromagram_norm = chromagram / chroma_norms
    
    # 4. Get Templates based on style
    templates, _ = generate_templates(style, transpose_semitones=0)
    # Normalize templates
    temp_norms = np.linalg.norm(templates, axis=1)
    templates_norm = templates / temp_norms[:, None]
    
    # 5. Template Matching (Cosine Similarity)
    similarity = 1 - cdist(chromagram_norm.T, templates_norm, metric='cosine')
    
    # Get highest matching chord for each frame
    raw_chord_indices = np.argmax(similarity, axis=1)
    
    # 6. Smoothing (Median Filter to remove quick jumps)
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
    smoothed_indices = scipy.signal.medfilt(raw_chord_indices, kernel_size=smooth_kernel).astype(int)
    
    # 7. Convert frames to timestamps
    times = librosa.frames_to_time(np.arange(len(smoothed_indices)), sr=sr)
    
    # 8. Beat Tracking for Chord Chart
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    return smoothed_indices, beat_frames, times, (estimated_key_str, estimated_root, estimated_is_major)

def format_results(smoothed_indices, beat_frames, times, style='Pop', transpose_semitones=0, key_info=None):
    use_flats = False
    if key_info is not None:
        _, estimated_root, estimated_is_major = key_info
        transposed_root = (estimated_root + transpose_semitones) % 12
        if estimated_is_major:
            use_flats = MAJOR_KEYS[transposed_root][1]
        else:
            use_flats = MINOR_KEYS[transposed_root][1]
            
    _, labels = generate_templates(style, transpose_semitones, use_flats=use_flats)
    
    beat_chords = []
    for bf in beat_frames:
        if bf < len(smoothed_indices):
            # Clean up the 'maj' suffix for Pop style to just the root note (e.g. Cmaj -> C)
            c = labels[smoothed_indices[bf]]
            if style == 'Pop' and c.endswith('maj'):
                c = c[:-3]
            beat_chords.append(c)
            
    # Format into measures (assuming 4/4 time signature)
    measures = []
    for i in range(0, len(beat_chords), 4):
        beats = beat_chords[i:i+4]
        m_str = ""
        last_c = None
        for b in beats:
            if b != last_c:
                m_str += f"{b:<5}"
                last_c = b
            else:
                m_str += "     "
        measures.append(f"| {m_str.rstrip()} ".ljust(16))
        
    chart_lines = []
    for i in range(0, len(measures), 4):
        chart_lines.append("".join(measures[i:i+4]) + "|")
    chart_string = "\n".join(chart_lines)
    
    # Group consecutive identical chords for table based on beats
    results = []
    
    if not beat_chords:
        return results, chart_string

    current_chord = beat_chords[0]
    start_time = times[0]  # Force the first segment to start at time 0
    
    for i in range(1, len(beat_chords)):
        chord = beat_chords[i]
        if chord != current_chord:
            results.append({
                "start": float(start_time),
                "end": float(times[beat_frames[i]]),
                "chord": current_chord
            })
            current_chord = chord
            start_time = times[beat_frames[i]]
            
    results.append({
        "start": float(start_time),
        "end": float(times[-1]),
        "chord": current_chord
    })
    
    return results, chart_string

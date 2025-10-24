import sqlite3
from pathlib import Path
import numpy as np
import torch
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent
from transformers import pipeline
import re

# --- Constants ---
DB_NAME = "test_tts_data.db"
TABLE_NAME = "processed_data"
ASR_MODEL_ID = "openai/whisper-medium"

# --- Segmentation Parameters ---
MIN_DURATION_MS = 3 * 1000
MAX_DURATION_MS = 15 * 1000
SILENCE_THRESHOLD_DB = -35
MIN_SILENCE_LEN_MS = 300  # A pause of this length is a potential split point
SEGMENT_PADDING_MS = 200  # Add padding to avoid cutting off words

# --- Quality Filter Parameters ---
# Audio Quality
MIN_RMS = 250               # Minimum average volume (RMS). Rejects whispers/silence.
MAX_CLIPPING_PERCENT = 1.0  # Max % of samples allowed to be "clipped" (distorted).

# Music Heuristic
MUSIC_BASS_HZ = 80          # Ignore deep room rumble below 80Hz.
MUSIC_TREBLE_HZ = 8000        # Frequencies above this are likely cymbals/hiss.
MUSIC_ENERGY_RATIO = 2.0    # Allow non-speech energy to be 2x speech energy.

# ASR Sanity Check
MAX_ASR_INPUT_MS = 29.5 * 1000  # Max duration to feed into Whisper (just under 30s)

# --- Model and Device Setup ---
try:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    ASR_PIPELINE = pipeline(
        "automatic-speech-recognition", model=ASR_MODEL_ID, device=DEVICE
    )
    print(f"Successfully loaded ASR model '{ASR_MODEL_ID}'.")
except Exception as e:
    print(f"Error loading ASR model: {e}")
    ASR_PIPELINE = None


def create_db(refresh: bool = True):
    """Create a SQLite DB. If refresh is True, drops the existing table."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        if refresh:
            print("Refreshing database: dropping existing table.")
            cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT NOT NULL,
                wav_path TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                rms REAL,
                clipping_percent REAL,
                music_ratio REAL,
                overlap_flag BOOLEAN DEFAULT 0
            )
        ''')
        conn.commit()
        print(f"Database '{DB_NAME}' and table '{TABLE_NAME}' are set up.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()


def create_batch(wav_paths):
    """Generator that loads, normalizes, and resamples one WAV file at a time."""
    for wav_path in wav_paths:
        try:
            print(f"\nLoading '{wav_path.name}'...")
            audio = AudioSegment.from_wav(wav_path)
            # --- Normalization Step ---
            # This brings the audio to a standard loudness before segmentation.
            audio = normalize(audio)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Resample to 16kHz for Whisper
            yield {'original_name': wav_path.name, 'audio': audio}
        except Exception as e:
            print(f"Could not process file {wav_path}: {e}")


def run_segmentation(batch):
    """
    A robust, two-pass segmentation algorithm. First, it splits any long
    utterances, then it merges smaller utterances to an optimal length.
    """
    audio = batch['audio']
    original_name = batch['original_name']
    print(f"Segmenting '{original_name}'...")

    nonsilent_intervals = detect_nonsilent(
        audio, min_silence_len=MIN_SILENCE_LEN_MS, silence_thresh=SILENCE_THRESHOLD_DB
    )
    if not nonsilent_intervals:
        print(f"  -> No speech detected in '{original_name}'.")
        return []

    # --- Pass 1: Split long chunks ---
    split_intervals = []
    for start_ms, end_ms in nonsilent_intervals:
        duration_ms = end_ms - start_ms
        if duration_ms > MAX_DURATION_MS:
            # This chunk is too long, split it aggressively
            for i in range(0, duration_ms, MAX_DURATION_MS):
                split_start = start_ms + i
                split_end = min(start_ms + i + MAX_DURATION_MS, end_ms)
                split_intervals.append((split_start, split_end))
        else:
            split_intervals.append((start_ms, end_ms))

    # --- Pass 2: Merge short chunks ---
    final_intervals = []
    if not split_intervals: return []  # Guard against empty list

    def save_if_valid(start_ms, end_ms):
        """Checks duration and appends to final_intervals."""
        duration = end_ms - start_ms
        if duration >= MIN_DURATION_MS:
            final_intervals.append((start_ms, end_ms))

    temp_segment_start = split_intervals[0][0]
    temp_segment_end = split_intervals[0][1]

    for i in range(1, len(split_intervals)):
        next_start, next_end = split_intervals[i]

        if next_end - temp_segment_start <= MAX_DURATION_MS:
            temp_segment_end = next_end
        else:
            save_if_valid(temp_segment_start, temp_segment_end)
            temp_segment_start = next_start
            temp_segment_end = next_end

    save_if_valid(temp_segment_start, temp_segment_end)

    # --- Finalization: Create AudioSegment objects with padding ---
    final_segments = []
    for start_ms, end_ms in final_intervals:
        start_with_padding = max(0, start_ms - SEGMENT_PADDING_MS)
        end_with_padding = min(len(audio), end_ms + SEGMENT_PADDING_MS)

        segment_audio = audio[start_with_padding:end_with_padding]
        final_segments.append({
            'original_name': original_name,
            'audio': segment_audio,
            'start_ms': start_ms,
            'end_ms': end_ms
        })

    print(f"  -> Segmented into {len(final_segments)} valid clips.")
    return final_segments


def run_audio_quality_filter(batch):
    """
    Calculate audio-only quality metrics and filter segments before feeding them to the ASR model.
    Metrics are saved to the segment dictionary and finally put in the SQL table.
    """
    if not batch:
        return []

    print(f"  -> Audio-filtering {len(batch)} segments pre-ASR...")
    filtered_batch = []

    for s in batch:
        audio = s['audio']

        # Calculate and Store Metrics
        s['rms'] = audio.rms

        # Calculate Clipping
        samples_array = audio.get_array_of_samples()
        samples = np.array(samples_array).astype(np.float32)
        if samples_array.typecode != 'f':
            samples /= np.iinfo(samples_array.typecode).max
        s['clipping_percent'] = np.mean(np.abs(samples) >= 0.98) * 100

        # Calculate Music Ratio
        try:
            calc_audio = audio.high_pass_filter(100).low_pass_filter(7500)

            total_rms = calc_audio.rms
            # Use a floor for RMS to avoid division by zero on pure silence
            if total_rms < 1: total_rms = 1

            bass_rms = calc_audio.low_pass_filter(MUSIC_BASS_HZ).rms
            treble_rms = calc_audio.high_pass_filter(MUSIC_TREBLE_HZ).rms

            non_speech_energy = (bass_rms + treble_rms)
            # Ensure speech_energy isn't negative if noise is overwhelming
            speech_energy = max(1, total_rms - non_speech_energy)

            s['music_ratio'] = non_speech_energy / speech_energy

        except Exception as e:
            print(f"    - WARNING: Error during music check: {e}")
            s['music_ratio'] = -1.0  # Use -1 as an error code

        # --- Filters ---
        passed = True  # Assume it passes unless a check fails

        if s['rms'] < MIN_RMS:
            print(f"    - REJECT (RMS): {s['rms']:.2f} < {MIN_RMS}")
            passed = False

        if s['clipping_percent'] > MAX_CLIPPING_PERCENT:
            print(f"    - REJECT (Clip): {s['clipping_percent']:.2f}% > {MAX_CLIPPING_PERCENT}%")
            passed = False

        if s['music_ratio'] > MUSIC_ENERGY_RATIO:
            print(f"    - REJECT (Music): {s['music_ratio']:.2f} > {MUSIC_ENERGY_RATIO}")
            passed = False

        if s['music_ratio'] == -1.0:  # Check for the error code
            print(f"    - REJECT (Music Error): Calculation failed.")
            passed = False


        if passed:
            # If it passed all audio checks, keep it
            filtered_batch.append(s)
            print(f"    - PASS: RMS={s['rms']:.2f}, Clip={s['clipping_percent']:.2f}%, Music={s['music_ratio']:.2f}")

    print(f"  -> Kept {len(filtered_batch)} segments after audio filtering.")
    return filtered_batch


def run_asr(batch):
    """Transcribe audio segments in a batch for efficiency."""
    if not ASR_PIPELINE or not batch:
        return []

    print(f"  -> Transcribing {len(batch)} segments in a batch...")

    # Prepare all audio segments for the pipeline
    audio_inputs = []
    for segment_data in batch:
        audio_segment = segment_data['audio']
        if len(audio_segment) > MAX_ASR_INPUT_MS:
            print(f"    - WARNING: Skipping segment over 29.5s.")
            continue

        # Get the raw samples
        samples_array = audio_segment.get_array_of_samples()
        samples = np.array(samples_array).astype(np.float32)

        # Only normalize if the samples are not already 32-bit floats
        if samples_array.typecode != 'f':
            max_val = np.iinfo(samples_array.typecode).max
            samples /= max_val

        audio_inputs.append({"sampling_rate": audio_segment.frame_rate, "raw": samples})

    # Run ASR pipeline on the entire batch
    # Adding batch_size=8 for a potential performance boost on GPU
    results = ASR_PIPELINE(
        audio_inputs, batch_size=8, generate_kwargs={"language": "english"}
    )

    # Assign transcriptions back to the batch
    for i, segment_data in enumerate(batch):
        segment_data['text'] = results[i]['text'].strip()
        print(f"    - Raw TXT {i}: {segment_data['text']}")

    return batch


def run_text_quality_filter(batch):
    """
    Apply text-based quality filters to the transcription, and check for word overlaps.
    """
    if not batch:
        return []

    print(f"  -> Text-filtering {len(batch)} segments post-ASR...")
    filtered_batch = []

    hallucination_pattern = re.compile(
        r'\[.*?\]|\(.*?\)|thanks for watching|thank you for watching',
        re.IGNORECASE
    )

    for s in batch:
        # Default the flag to False
        s['overlap_flag'] = False
        text = s['text']

        # --- Text Quality Checks ---
        if not text or len(text.split()) <= 2:
            continue
        if not re.search(r'[a-zA-Z]', text):
            continue
        if hallucination_pattern.search(text):
            continue

        filtered_batch.append(s)

    # --- Overlap Check Pass ---
    print(f"  -> Checking {len(filtered_batch)} segments for word overlap...")
    for i in range(1, len(filtered_batch)):
        # Only compare segments from the same original file
        if filtered_batch[i]['original_name'] != filtered_batch[i - 1]['original_name']:
            continue

        try:
            prev_text = filtered_batch[i - 1]['text'].lower().split()
            curr_text = filtered_batch[i]['text'].lower().split()

            if not prev_text or not curr_text:
                continue

            # If last word of previous clip == first word of current clip
            if prev_text[-1] == curr_text[0]:
                filtered_batch[i - 1]['overlap_flag'] = True
                filtered_batch[i]['overlap_flag'] = True
        except Exception as e:
            print(f"    - WARNING: Error during overlap check: {e}")

    print(f"  -> Kept {len(filtered_batch)} high-quality segments after text filtering.")
    return filtered_batch

def save_processed_audio(batch, out_dir):
    """Save the processed audio segments to disk."""
    print(f"  -> Saving {len(batch)} audio segments to '{out_dir}'...")
    for i, segment_data in enumerate(batch):
        original_stem = Path(segment_data['original_name']).stem
        start_s = segment_data['start_ms'] // 1000
        end_s = segment_data['end_ms'] // 1000
        output_filename = f"{original_stem}_{start_s:04d}s_{end_s:04d}s.wav"
        output_path = out_dir / output_filename

        try:
            segment_data['audio'].export(output_path, format="wav")
            segment_data['wav_path'] = output_path
        except Exception as e:
            print(f"    -> ERROR: Could not save file {output_path}: {e}")
            segment_data['wav_path'] = None

    return [s for s in batch if s.get('wav_path') is not None]

def add_batch_metadata_to_db(batch):
    """Add metadata from a processed batch to the database."""
    if not batch:
        print("  -> No data to add to the database.")
        return

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Prepare data, using .get() for safety in case a key is missing
        data_to_insert = [
            (
                item['original_name'],
                str(item['wav_path']),
                item['text'],
                item.get('rms'),
                item.get('clipping_percent'),
                item.get('music_ratio'),
                item.get('overlap_flag', False)
            ) for item in batch
        ]

        cursor.executemany(f'''
            INSERT OR IGNORE INTO {TABLE_NAME} (
                original_name, wav_path, text, 
                rms, clipping_percent, music_ratio, overlap_flag
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)

        conn.commit()
        print(f"  -> Successfully added {len(data_to_insert)} records to the database.")
    except sqlite3.Error as e:
        print(f"  -> Database error during insertion: {e}")
    finally:
        if conn:
            conn.close()

def run_pipeline(wav_dir, out_dir):
    """Run the full audio processing pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(list(wav_dir.glob("*.wav")))
    if not wav_paths:
        print(f"No .wav files found in '{wav_dir}'. Please add some audio files.")
        return

    create_db(refresh=True)

    total_final_segments = 0
    for file_batch in create_batch(wav_paths):
        # 1. Segment audio
        segmented_batch = run_segmentation(file_batch)

        # 2. Run audio quality filter
        audio_filtered_batch = run_audio_quality_filter(segmented_batch)

        # 3. Run ASR
        transcribed_batch = run_asr(audio_filtered_batch)

        # 4. Run text quality filter
        text_filtered_batch = run_text_quality_filter(transcribed_batch)

        # 5. Save the final, clean batch
        saved_batch = save_processed_audio(text_filtered_batch, out_dir)

        if saved_batch:
            add_batch_metadata_to_db(saved_batch)
            total_final_segments += len(saved_batch)

    print("\n--- Pipeline Finished ---")
    print(f"Processed {len(wav_paths)} audio file(s).")
    print(f"Saved {total_final_segments} high-quality segments to '{out_dir}'.")

if __name__ == "__main__":
    raw_data_dir = Path("data")
    raw_data_dir.mkdir(exist_ok=True)
    out_dir = Path("processed_audio")
    run_pipeline(raw_data_dir, out_dir)


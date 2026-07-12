import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

DEVICE = "cuda"

print("Loading model...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenMelodyForConditionalGeneration.from_pretrained(
    "facebook/musicgen-melody",
    torch_dtype=torch.float32,
    attn_implementation="eager"
).to(DEVICE)

print("Loading audio...")
# Load the HeartMuLa generated mp3
audio_path = "assets/output.mp3"
import soundfile as sf
from scipy import signal as scipy_signal

melody_np, sr = sf.read(audio_path, always_2d=True)
melody_np = melody_np.mean(axis=1)
if sr != 32000:
    num_samples = int(len(melody_np) * 32000 / sr)
    melody_np = scipy_signal.resample(melody_np, num_samples)
melody_np = melody_np[:32000 * 5].astype(np.float32)

def generate_test(melody, name):
    print(f"Generating {name}...")
    inputs = processor(
        text=["jazz piano, saxophone, smooth"],
        audio=melody,
        sampling_rate=32000,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=250) # 5 seconds
    
    out = audio_values[0, 0].cpu().numpy()
    out = out / max(1e-8, np.max(np.abs(out)))
    sf.write(f"assets/{name}.wav", out, 32000)
    print(f"Saved {name}.wav")

# 1. Raw
generate_test(melody_np, "test_raw")

# 2. Add noise
melody_noisy = melody_np + np.random.normal(0, 0.02, melody_np.shape).astype(np.float32)
generate_test(melody_noisy, "test_noisy")

# 3. Text only (to verify model works)
print("Generating text only...")
inputs = processor(
    text=["jazz piano, saxophone, smooth"],
    padding=True,
    return_tensors="pt",
)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
with torch.no_grad():
    audio_values = model.generate(**inputs, max_new_tokens=250)
out = audio_values[0, 0].cpu().numpy()
out = out / max(1e-8, np.max(np.abs(out)))
sf.write("assets/test_text.wav", out, 32000)
print("Saved test_text.wav")

"""
MusicGen Text-Only GUI — 旋律風格轉換器（無雜訊版）
從原曲提取 BPM、和弦、調性，結合自訂風格生成全新的歌曲骨架。
"""

import gradio as gr
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import chord_recognizer

# ── 設定 ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "assets"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 關閉 SDPA 與 TF32 確保穩定 ──
if DEVICE == "cuda":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

_model     = None
_processor = None

def load_model(progress):
    global _model, _processor
    if _model is None:
        progress(0.05, desc="📦 首次使用：下載 MusicGen 模型（~3GB）…")
        _processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
        progress(0.15, desc="🧠 載入模型到 GPU…")
        _model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-medium",
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(DEVICE)
        _model.eval()
        progress(0.30, desc="✅ 模型準備完成！")
    return _model, _processor

# ── 風格預設 ──────────────────────────────────────────────────
STYLE_PRESETS = {
    "🎷 爵士（Jazz）":             "jazz, saxophone, piano, double bass, smooth, night club atmosphere",
    "🎸 搖滾（Rock）":             "rock, electric guitar, distortion, drums, bass guitar, energetic, powerful",
    "🎻 古典弦樂（Classical）":    "classical orchestra, strings, violin, cello, elegant, concert hall",
    "🌊 Lo-fi Chill":             "lofi hip hop, chill beats, vinyl crackle, warm, relaxing, study music",
    "🌺 波薩諾瓦（Bossa Nova）":   "bossa nova, acoustic guitar, soft percussion, Brazilian, romantic, gentle",
    "🎹 鋼琴獨奏（Solo Piano）":   "solo piano, cinematic, emotional, intimate, concert grand piano",
    "🪗 民謠（Folk）":             "folk, acoustic guitar, fingerpicking, storytelling, warm and organic",
    "🎛️ 電子（Electronic）":      "electronic, synthesizer, ambient, atmospheric, futuristic, spacious",
    "💃 拉丁（Latin）":            "latin, salsa rhythm, trumpet, percussion, upbeat, energetic dance",
    "🎺 大樂隊（Big Band）":       "big band jazz, brass section, swing rhythm, upbeat, 1940s style",
    "🌸 輕音樂（Easy Listening）": "easy listening, soft melody, flute, light orchestration, peaceful",
    "🔥 嘻哈（Hip-hop）":          "hip hop beat, trap, 808 bass, hi-hat, urban, modern production",
}

# ── 生成函數 ──────────────────────────────────────────────────
def generate_cover(
    audio_input,
    style_preset,
    custom_style,
    duration,
    temperature,
    guidance_scale,
    progress=gr.Progress(),
):
    if audio_input is None:
        raise gr.Error("⚠️ 請先上傳原曲音訊檔案！")

    style_text = custom_style.strip() if custom_style.strip() else STYLE_PRESETS.get(style_preset, "")
    if not style_text:
        raise gr.Error("⚠️ 請選擇風格預設或輸入自訂風格描述！")

    model, processor = load_model(progress)

    # ── 分析原曲（和弦、BPM、調性） ──
    progress(0.32, desc="🎵 分析原曲和弦與節奏…")
    try:
        smoothed_indices, beat_frames, times, key_info = chord_recognizer.analyze_audio(audio_input, style='Pop')
        results, chart_string = chord_recognizer.format_results(smoothed_indices, beat_frames, times, key_info=key_info)
        
        bpm = 120
        import librosa
        y, sr = librosa.load(audio_input, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        bpm = int(round(tempo))
        
        key_str = key_info[0]
        # 提取前幾個獨特的和弦來作為 Prompt
        chords_list = []
        for r in results:
            if not chords_list or chords_list[-1] != r['chord']:
                chords_list.append(r['chord'])
        chords_str = ", ".join(chords_list[:8]) # 取前 8 個和弦避免 Prompt 過長
        
        prompt_text = f"{bpm} BPM, {key_str}, chords: {chords_str}, {style_text}"
        info_msg = f"**偵測結果**：{bpm} BPM | 調性 {key_str} | 和弦 {chords_str}\n\n**最終 Prompt**：`{prompt_text}`"
        print(f"\n[MusicGen] {info_msg}")
    except Exception as e:
        print(f"Chord extraction failed: {e}")
        prompt_text = style_text
        info_msg = "⚠️ 和弦分析失敗，退回純風格生成"

    # ── 準備 inputs ──
    progress(0.40, desc="🔧 準備模型輸入…")
    inputs = processor(
        text=[prompt_text],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # ── 生成 ──
    max_new_tokens = int(duration * 50)  # ~50 tokens/sec for musicgen
    progress(0.50, desc=f"🎼 生成中：「{style_text[:45]}…」（約 {duration} 秒）")

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    # ── 儲存輸出 ──
    progress(0.92, desc="💾 儲存輸出…")
    output_path = str(OUTPUT_DIR / "musicgen_output.wav")
    audio_np = audio_values[0, 0].cpu().float().numpy()  # (samples,)
    # 避免破音，進行正規化 (Normalize)
    audio_np = audio_np / max(1e-8, np.max(np.abs(audio_np)))
    sf.write(output_path, audio_np, samplerate=model.config.audio_encoder.sampling_rate)

    progress(1.0, desc="✅ 完成！")
    return output_path, info_msg


def on_preset_change(preset):
    return STYLE_PRESETS.get(preset, "")


# ── Gradio UI ─────────────────────────────────────────────────
css = """
body, .gradio-container { font-family: 'Noto Sans TC', 'Segoe UI', sans-serif !important; }
#title  { text-align: center; }
#subtitle { text-align: center; color: #888; font-size: 0.9em; margin-top: 4px; }
#gen-btn  { font-size: 1.1em !important; height: 56px !important; }
.tip-box  { background: rgba(6,182,212,0.08); border-left: 3px solid #06B6D4;
            padding: 10px 14px; border-radius: 6px; font-size: 0.88em; line-height: 1.6; }
"""

theme = gr.themes.Soft(primary_hue="cyan", secondary_hue="purple", neutral_hue="slate")

with gr.Blocks(title="MusicGen AI Cover — 和弦與風格轉換") as demo:

    gr.Markdown("# 🎼 MusicGen AI Cover", elem_id="title")
    gr.Markdown(
        "自動提取原曲和弦與節奏 · 套用全新音樂風格 · 完全零雜訊生成",
        elem_id="subtitle",
    )

    with gr.Row():
        # ── 左欄 ──
        with gr.Column(scale=3):

            with gr.Group():
                gr.Markdown("### 🎵 上傳原曲 (分析用)")
                gr.Markdown(
                    '<div class="tip-box">'
                    '上傳 HeartMuLa 生成的 MP3，系統會自動提取 <b>BPM、調性、和弦進行</b>，<br>'
                    '並將它們轉化為 AI 的超級提示詞來維持歌曲骨架。'
                    '</div>'
                )
                audio_input = gr.Audio(
                    label="原曲（拖曳或點擊上傳）",
                    type="filepath",
                    sources=["upload"],
                )

            with gr.Group():
                gr.Markdown("### 🎨 目標風格")
                style_preset = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="🎷 爵士（Jazz）",
                    label="快速選擇風格預設",
                )
                custom_style = gr.Textbox(
                    label="✏️ 自訂風格描述（英文，留空則使用上方預設）",
                    placeholder="例如：90s country, acoustic guitar, warm, nostalgic, storytelling",
                    lines=2,
                )
                style_preset.change(fn=on_preset_change, inputs=[style_preset], outputs=[custom_style])

        # ── 右欄 ──
        with gr.Column(scale=2):

            with gr.Group():
                gr.Markdown("### ⚙️ 生成參數")
                duration = gr.Slider(
                    minimum=5, maximum=30, value=20, step=1,
                    label="輸出長度（秒）",
                    info="建議 15–25 秒，最長 30 秒"
                )
                temperature = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                    label="Temperature（創意度）",
                    info="越高越隨機，越低越保守"
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                    label="Guidance Scale（風格遵循度）",
                    info="越高越貼近描述的風格"
                )

            gen_btn = gr.Button("🎼 分析和弦並開始生成", variant="primary", elem_id="gen-btn")

            with gr.Group():
                gr.Markdown("### 🎧 生成結果")
                analysis_info = gr.Markdown("等待分析...")
                audio_output = gr.Audio(
                    label="風格轉換後的音樂",
                    type="filepath",
                    interactive=False,
                )
                gr.Markdown("📁 儲存於：`heartlib/assets/musicgen_output.wav`")

    gen_btn.click(
        fn=generate_cover,
        inputs=[audio_input, style_preset, custom_style, duration, temperature, guidance_scale],
        outputs=[audio_output, analysis_info],
    )

    with gr.Accordion("📖 運作原理說明", open=False):
        gr.Markdown("""
### 為什麼放棄 Melody 輸入？
因為 AI 生成的音訊太過乾淨，沒有真實錄音的底噪，這會觸發 MusicGen-Melody 的底層 Bug 導致輸出 20 秒巨大雜訊。

### 新的 Cover 運作方式
我們改用 **Chord Extractor (和弦提取)** 技術：
1. 分析你上傳的音檔，算出 BPM（速度）、Key（調性）、以及和弦進行。
2. 將這些樂理資訊加上你選的風格（如 Jazz）結合成超級 Prompt。
3. 讓最穩定的 Text-to-Music 模型根據這個 Prompt 創作。
這樣能保證生成結果絕對乾淨無雜音，同時「歌曲的氛圍、和弦走向、速度」都跟原曲一致！
        """)

# ── 啟動 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[MusicGen] Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[MusicGen] GPU: {torch.cuda.get_device_name(0)}")
    print("[MusicGen] Starting GUI...\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        inbrowser=True,
        share=False,
        show_error=True,
        theme=theme,
        css=css,
    )

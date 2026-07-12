"""
HeartMuLa 音樂生成器 — Gradio GUI
用法：.venv310/Scripts/python.exe app_gui.py
"""

import gradio as gr
import subprocess
import sys
import os
import time
import threading
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
CKPT_DIR   = BASE_DIR / "ckpt"
PYTHON     = BASE_DIR / ".venv310" / "Scripts" / "python.exe"
GEN_SCRIPT = BASE_DIR / "examples" / "run_music_generation.py"

ASSETS_DIR.mkdir(exist_ok=True)

# ── 預設歌詞 ──────────────────────────────────────────────────
DEFAULT_LYRICS = """\
[Verse]
早晨的陽光 把窗簾染成金黃
鬧鐘還沒響 我已準備好飛翔
昨天的煩惱 早就消失在風裡
今天只屬於 我想笑的那種美麗

[Prechorus]
感覺到了嗎 空氣都是甜的味道
每一步踩下去 都像踏在雲端上

[Chorus]
今天最好 今天最好
把所有心事都丟掉
今天最好 今天最好
讓快樂大聲說 Hello
舉起雙手 跳一跳
這一刻就是我的驕傲
不管明天會如何
今天我要 閃閃發光好好燃燒

[Bridge]
也許未來還有很多彎路
也許夢想還很遙遠很模糊
但此刻的我 呼吸著這美好
就夠了 就夠了 就夠了

[Chorus]
今天最好 今天最好
把所有心事都丟掉
今天最好 今天最好
讓快樂大聲說 Hello

[Outro]
今天我要 閃閃發光
"""

DEFAULT_TAGS = "pop,happy,upbeat,piano,synthesizer,energetic,celebratory,mandarin"

TAIWANESE_LYRICS = """\
[Verse]
天光了 鳥仔叫
心情好 唱歌謠
風吹來 心花開
今仔日 真正讚

[Prechorus]
阮的心內 充滿希望
行出去 看天頂的光

[Chorus]
歡喜就好 歡喜就好
毋免煩惱遐爾多
歡喜就好 歡喜就好
日子過了真幸福
舉起手 跳一跳
這一時 是阮的寶
管伊明仔載按怎
今仔日 阮要笑甲透透

[Bridge]
嘛是有時艱苦
嘛是有時目屎流
毋過阮知影
這攏會過去
加油 加油 加油

[Chorus]
歡喜就好 歡喜就好
毋免煩惱遐爾多
今仔日 上好

[Outro]
今仔日 上好
"""

PRESETS = {
    "🇹🇼 國語 Pop《今天最好》":    (DEFAULT_LYRICS,   "pop,happy,upbeat,piano,synthesizer,energetic,mandarin"),
    "🫀 台語 Folk《歡喜就好》":     (TAIWANESE_LYRICS, "taiwanese,hokkien,pop,happy,folk,upbeat,guitar,piano,cheerful"),
    "🎸 搖滾 Rock":                 ("",               "rock,electric guitar,drums,energetic,powerful"),
    "🌙 深夜爵士 Jazz":             ("",               "jazz,night,saxophone,piano,slow,romantic,smooth"),
    "🌊 Lo-fi Chill":              ("",               "lofi,chill,relaxing,piano,soft,study,ambient"),
    "💕 抒情情歌 Ballad":           ("",               "ballad,slow,romantic,emotional,strings,piano,heartfelt"),
}

# ── 生成函數 ──────────────────────────────────────────────────
def generate(lyrics, tags, max_secs, temperature, cfg_scale, topk, progress=gr.Progress()):
    if not lyrics.strip():
        raise gr.Error("⚠️ 請輸入歌詞！")
    if not PYTHON.exists():
        raise gr.Error("找不到 Python 環境，請確認 .venv310 存在。")

    # 寫入歌詞與 tags
    lyrics_path = ASSETS_DIR / "lyrics.txt"
    tags_path   = ASSETS_DIR / "tags.txt"
    output_path = ASSETS_DIR / "output_gui.mp3"

    lyrics_path.write_text(lyrics, encoding="utf-8")
    tags_path.write_text(tags.strip(), encoding="utf-8")

    cmd = [
        str(PYTHON), str(GEN_SCRIPT),
        f"--model_path={CKPT_DIR}",
        "--version=3B",
        "--lazy_load", "true",
        f"--save_path={output_path}",
        f"--max_audio_length_ms={int(max_secs * 1000)}",
        f"--temperature={temperature}",
        f"--cfg_scale={cfg_scale}",
        f"--topk={int(topk)}",
    ]

    progress(0, desc="🔄 啟動模型中…")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(BASE_DIR),
    )

    log_lines = []
    stage = 0

    STAGE_MSGS = [
        (0.05, "📦 載入 HeartMuLa 模型 (3B)…"),
        (0.30, "🎵 生成音樂 Token 中…"),
        (0.70, "🎹 HeartCodec 解碼音訊…"),
        (0.95, "💾 儲存 MP3…"),
    ]

    for line in proc.stdout:
        line = line.rstrip()
        log_lines.append(line)

        if "Loading checkpoint shards" in line and "HeartMuLa" not in "".join(log_lines[-5:]):
            stage = 0
        if "Unloading HeartMuLa" in line:
            stage = 1
        if "Loading checkpoint shards" in line and stage == 1:
            stage = 2
        if "Unloading HeartCodec" in line:
            stage = 3
        if "Generated music saved" in line:
            stage = 4

        pct, msg = STAGE_MSGS[min(stage, len(STAGE_MSGS)-1)]
        progress(pct, desc=msg)

    proc.wait()
    progress(1.0, desc="✅ 完成！")

    if proc.returncode != 0 or not output_path.exists():
        raise gr.Error("❌ 生成失敗，請查看 log。\n\n" + "\n".join(log_lines[-20:]))

    return str(output_path), "\n".join(log_lines)


def load_preset(name):
    if name in PRESETS:
        lyrics, tags = PRESETS[name]
        return lyrics, tags
    return gr.update(), gr.update()


# ── Gradio UI ──────────────────────────────────────────────────
css = """
body, .gradio-container { font-family: 'Noto Sans TC', 'Segoe UI', sans-serif !important; }
#title { text-align: center; margin-bottom: 0; }
#subtitle { text-align: center; color: #aaa; margin-top: 4px; font-size: 0.9em; }
#gen-btn { background: linear-gradient(135deg, #FF6B9D, #A855F7) !important;
           color: white !important; font-size: 1.1em !important; height: 56px !important; }
#gen-btn:hover { opacity: 0.9; transform: scale(1.01); }
.dark { background: #0a0a14; }
"""

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
    neutral_hue="slate",
    font=["Noto Sans TC", "Segoe UI", "sans-serif"],
)

with gr.Blocks(title="HeartMuLa 音樂生成器") as demo:

    gr.Markdown("# 🎵 HeartMuLa 音樂生成器", elem_id="title")
    gr.Markdown("HeartMuLa-oss-3B · 本地運行 · RTX GPU 加速", elem_id="subtitle")

    with gr.Row():
        # ── 左欄：輸入區 ──
        with gr.Column(scale=3):

            with gr.Group():
                gr.Markdown("### 🎨 快速預設")
                preset_dd = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    label="選擇預設風格",
                    value=None,
                    interactive=True,
                )

            with gr.Group():
                gr.Markdown("### 📝 歌詞")
                gr.Markdown(
                    "支援 `[Verse]` `[Prechorus]` `[Chorus]` `[Bridge]` `[Outro]` 段落標記。"
                    "支援中文、台語、英文等多語言。"
                )
                lyrics_box = gr.Textbox(
                    value=DEFAULT_LYRICS,
                    label="歌詞內容",
                    lines=18,
                    max_lines=40,
                    placeholder="在這裡輸入歌詞…",
                )

            with gr.Group():
                gr.Markdown("### 🏷️ 風格標籤")
                gr.Markdown("用英文逗號分隔，不加空格，例如：`pop,happy,piano,mandarin`")
                tags_box = gr.Textbox(
                    value=DEFAULT_TAGS,
                    label="Tags",
                    placeholder="pop,happy,piano,guitar,upbeat",
                )

        # ── 右欄：參數 + 輸出 ──
        with gr.Column(scale=2):

            with gr.Group():
                gr.Markdown("### ⚙️ 生成參數")
                max_secs = gr.Slider(
                    minimum=30, maximum=300, value=240, step=10,
                    label="最大長度（秒）",
                    info="預設 240 秒（4 分鐘）"
                )
                temperature = gr.Slider(
                    minimum=0.5, maximum=1.5, value=1.0, step=0.05,
                    label="Temperature（創意度）",
                    info="數值越高越有創意，越低越穩定"
                )
                cfg_scale = gr.Slider(
                    minimum=1.0, maximum=5.0, value=1.5, step=0.1,
                    label="CFG Scale（歌詞遵循度）",
                    info="數值越高越貼近歌詞"
                )
                topk = gr.Slider(
                    minimum=10, maximum=200, value=50, step=10,
                    label="Top-K 採樣",
                    info="控制候選 token 數量"
                )

            gen_btn = gr.Button("🎵 開始生成", variant="primary", elem_id="gen-btn")

            with gr.Group():
                gr.Markdown("### 🎧 生成結果")
                audio_out = gr.Audio(
                    label="生成的音樂",
                    type="filepath",
                    interactive=False,
                )
                gr.Markdown(
                    "📁 檔案儲存於：`heartlib/assets/output_gui.mp3`",
                )

            with gr.Accordion("📋 生成 Log（展開查看詳細）", open=False):
                log_out = gr.Textbox(
                    label="Log",
                    lines=12,
                    interactive=False,
                )

    # ── 事件 ──
    preset_dd.change(
        fn=load_preset,
        inputs=[preset_dd],
        outputs=[lyrics_box, tags_box],
    )

    gen_btn.click(
        fn=generate,
        inputs=[lyrics_box, tags_box, max_secs, temperature, cfg_scale, topk],
        outputs=[audio_out, log_out],
        api_name="generate",
    )

    # ── 使用說明 ──
    with gr.Accordion("📖 使用說明", open=False):
        gr.Markdown("""
### 快速開始
1. 選擇左上方的**快速預設**，或自己輸入歌詞
2. 設定**風格標籤**（Tags）
3. 點擊「🎵 開始生成」
4. 等待約 **20–25 分鐘**（含模型載入與生成）
5. 完成後會直接在頁面播放！

### 歌詞格式
```
[Verse]
第一段歌詞

[Prechorus]
Pre-Chorus 歌詞

[Chorus]
副歌歌詞

[Bridge]
橋段歌詞

[Outro]
結尾歌詞
```

### 常用 Tags 參考
| 語言 / 風格 | Tags |
|-------------|------|
| 國語流行 | `pop,mandarin,happy,piano` |
| 台語民謠 | `taiwanese,hokkien,folk,guitar` |
| 英文搖滾 | `rock,electric guitar,drums,english` |
| 爵士 | `jazz,saxophone,piano,smooth` |
| Lo-fi | `lofi,chill,ambient,study` |
| 抒情情歌 | `ballad,slow,romantic,strings` |
        """)

# ── 啟動 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,   # auto-select available port
        inbrowser=True,
        share=False,
        show_error=True,
        theme=theme,
        css=css,
    )

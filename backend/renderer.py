import os
import re
import sys
import math
import time
import json
import hashlib
import threading
import subprocess
import cv2
import numpy as np
from backend.utils import (
    log,
    format_ass_timestamp,
    format_timestamp,
    run_command,
    get_ffmpeg_binary,
    get_ffprobe_binary,
    TEMP_DIR,
    FONTS_DIR,
)

try:
    from backend.utils import env_float, env_bool
except ImportError:
    def env_float(key, default=0.0, min_val=None, max_val=None):
        val_str = os.environ.get(key)
        if val_str is None or str(val_str).strip() == "":
            return default
        try:
            val = float(val_str)
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)
            return val
        except (ValueError, TypeError):
            return default

    def env_bool(key, default=False):
        val_str = os.environ.get(key)
        if val_str is None or str(val_str).strip() == "":
            return default
        return str(val_str).strip().lower() in ("1", "true", "yes", "on", "y", "t")
from moviepy import VideoFileClip
from proglog import ProgressBarLogger

CASCADE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
PROFILE_CASCADE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_profileface.xml")
YUNET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx")
YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")

# ── SUBTITLE SAFE ZONE (video 9:16 = 1080x1920) ──
# Horizontal margin per side so subtitle text never touches the frame edge or
# collides with TikTok/Reels UI (right action rail, left text column).
# 120px ≈ 11% of the 1080px width per side -> text confined to the middle 78%.
SAFE_MARGIN_H = 120
# Vertical safe band (percent of height). Keeps text clear of the top header
# tabs (~12%) and the bottom caption + navigation bar (~20%).
SAFE_TOP_PCT = 12.0
SAFE_BOTTOM_PCT = 20.0
# WrapStyle 0 = smart word wrapping. Long lines wrap onto a second line inside
# the margins instead of overflowing past the frame edge (WrapStyle 1/2 clip).
SAFE_WRAP_STYLE = 0

# Per-style character budgets derived from the safe width (1080 - 2*120 = 840px)
# and each font's approximate average glyph width (~0.62 * font size).
STYLE_CHUNK_LIMITS = {
    "tiktok_viral": {"max_words": 4, "max_chars": 22},
    "classic_karaoke": {"max_words": 4, "max_chars": 22},
    "minimalist": {"max_words": 5, "max_chars": 24},
}

# Emoji tidak lagi dirender di subtitle (fitur emoji dihapus). Pola ini hanya
# dipakai untuk MEMBUANG emoji dari teks, karena font subtitle
# (Plus Jakarta Sans / Montserrat / Impact) tidak punya glyph emoji sehingga
# karakter emoji akan tampil sebagai kotak kosong (tofu).
_EMOJI_CHAR_RANGES = (
    r"\U0001F000-\U0001FAFF"      # emoticon, pictograph, transport, symbol
    r"\u2190-\u21FF"              # panah
    r"\u2300-\u23FF"              # misc technical
    r"\u2460-\u24FF"              # enclosed alphanumerics
    r"\u25A0-\u25FF"              # geometric shapes
    r"\u2600-\u27BF"              # misc symbols & dingbats
    r"\u2B00-\u2BFF"              # misc symbols and arrows
    r"\u3030\u303D\u3297\u3299"   # CJK symbols
    r"\u2122\u2139\u2049\u203C"   # trademark / info / punctuation emoji
)
_EMOJI_CLUSTER = (
    r"[\U0001F1E6-\U0001F1FF]{2}"
    rf"|[{_EMOJI_CHAR_RANGES}][\uFE0E\uFE0F]?[\U0001F3FB-\U0001F3FF]?"
    rf"(?:\u200D[{_EMOJI_CHAR_RANGES}][\uFE0E\uFE0F]?[\U0001F3FB-\U0001F3FF]?)*"
    r"|\u20E3"
)
_EMOJI_PATTERN = re.compile(f"({_EMOJI_CLUSTER})")

def _strip_emoji(text: str) -> str:
    """Buang seluruh emoji + variation selector dari teks subtitle."""
    if not text:
        return ""
    cleaned = _EMOJI_PATTERN.sub("", str(text))
    cleaned = cleaned.replace("\ufe0e", "").replace("\ufe0f", "").replace("\u200d", "")
    return re.sub(r"\s{2,}", " ", cleaned).strip()

def _escape_ass_text(text: str) -> str:
    """Escape ASS/SSA control characters and strip raw newlines."""
    if not text:
        return ""
    # Replace backslash first, then curly braces to prevent style tag injection
    sanitized = str(text).replace("\\", "＼").replace("{", "｛").replace("}", "｝")
    sanitized = sanitized.replace("\r", " ").replace("\n", " ").strip()
    return sanitized

def _display_len(text: str) -> int:
    """Panjang teks yang memakan lebar baris (emoji sudah dibuang lebih awal)."""
    if not text:
        return 0
    return len(_strip_emoji(text))

def _sanitize_word_timestamps(words, clip_start, clip_end, min_word_dur=0.18):
    """
    Sanitize and enforce strictly monotonic, non-overlapping timestamps.
    Clamps word relative start and end strictly within [0.0, clip_end - clip_start].
    Prevents subtitle overlapping, flickering, and jumbled text when speaker talks fast.
    """
    cleaned = []
    max_clip_dur = max(0.0, float(clip_end - clip_start))
    for w in words:
        w_start = max(0.0, min(max_clip_dur, float(w.get("start", 0.0)) - clip_start))
        w_end = max(0.0, min(max_clip_dur, float(w.get("end", 0.0)) - clip_start))
        text = _escape_ass_text(_strip_emoji(w.get("word", "")))
        if not text:
            continue
        if w_end <= w_start:
            w_end = min(max_clip_dur, w_start + min_word_dur)
        cleaned.append({"word": text, "start": w_start, "end": w_end})
        
    if not cleaned:
        return []
        
    cleaned.sort(key=lambda x: x["start"])
    
    # Monotonic clamping: ensure no word overlaps another
    sanitized = []
    for _i, w in enumerate(cleaned):
        s = max(0.0, min(max_clip_dur, w["start"]))
        e = max(0.0, min(max_clip_dur, max(w["end"], s + min_word_dur)))
        
        if sanitized and s < sanitized[-1]["end"]:
            s = sanitized[-1]["end"]
            e = max(0.0, min(max_clip_dur, max(e, s + min_word_dur)))
            
        sanitized.append({"word": w["word"], "start": s, "end": e})
        
    return sanitized

def _calc_margin_v(sub_position=68):
    """Konversi posisi persentase dari atas layar ke nilai ASS MarginV (dari bawah).

    Posisi di-clamp ke SAFE ZONE vertikal: subtitle tidak boleh masuk ke area
    header atas (SAFE_TOP_PCT) maupun area caption/navigation bawah
    (SAFE_BOTTOM_PCT). Karena style memakai Alignment 2 (bottom-anchored),
    ``pos`` merepresentasikan tepi BAWAH blok subtitle.
    """
    try:
        pos = float(sub_position)
    except (ValueError, TypeError):
        pos = 68.0
    min_pos = SAFE_TOP_PCT          # jangan sampai blok naik ke area header
    max_pos = 100.0 - SAFE_BOTTOM_PCT  # jangan sampai turun ke area caption/nav
    pos = max(min_pos, min(max_pos, pos))
    return int(1920 * ((100.0 - pos) / 100.0))

def _build_watermark_ass_section(clip_start, clip_end, source_watermark="", account_watermark=""):
    """Mengembalikan (styles_str, dialogues_list) untuk watermark Liquid Glass (teks kaca transparan tanpa background box)."""
    clean_src = (source_watermark or "").strip().replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ")
    clean_acc = (account_watermark or "").strip().replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ")
    if clean_acc and not clean_acc.startswith("@"):
        clean_acc = f"@{clean_acc}"

    styles = []
    dialogues = []
    duration = max(0.1, clip_end - clip_start)
    start_ass = "0:00:00.00"
    end_ass = format_ass_timestamp(duration)

    # Liquid Glass Text: BorderStyle=1 (tanpa background box/pill):
    # - Fill dalam kaca semi-transparan ({\1a&H65&}) sehingga warna video di belakang huruf menembus
    # - Rim kaca putih tajam ({\3c&HFFFFFF&\3a&H18&\bord1.3}) untuk efek bias cahaya tepi huruf
    # - Soft drop shadow ({\4c&H000000&\4a&H85&\shad2}) agar tetap terbaca di background terang maupun gelap
    if clean_src:
        # Style GlassSource: Top-Left (Alignment 7), margin_v=160, margin_l=55, BorderStyle 1 (no background box)
        styles.append("Style: GlassSource,Plus Jakarta Sans,34,&H65F8F4EE,&H000000FF,&H18FFFFFF,&H85000000,-1,0,0,0,100,100,1.2,0,1,1.3,2.0,7,55,55,160,1")
        src_text = f"Source: {clean_src}"
        dialogues.append(
            f"Dialogue: 0,{start_ass},{end_ass},GlassSource,,0,0,0,,"
            f"{{\\1c&HF8F4EE&\\1a&H65&\\3c&HFFFFFF&\\3a&H18&\\4c&H000000&\\4a&H85&\\bord1.3\\shad2.0\\blur0.6}}{src_text}"
        )

    if clean_acc:
        # Style GlassAccount: Top-Right (Alignment 9), margin_v=160, margin_r=55, BorderStyle 1 (no background box)
        styles.append("Style: GlassAccount,Plus Jakarta Sans,36,&H60F8F4EE,&H000000FF,&H18FFFFFF,&H85000000,-1,0,0,0,100,100,1.2,0,1,1.3,2.0,9,55,55,160,1")
        dialogues.append(
            f"Dialogue: 0,{start_ass},{end_ass},GlassAccount,,0,0,0,,"
            f"{{\\1c&HF8F4EE&\\1a&H60&\\3c&HFFFFFF&\\3a&H18&\\4c&H000000&\\4a&H85&\\bord1.3\\shad2.0\\blur0.6}}{clean_acc}"
        )

    return "\n".join(styles), dialogues


def _generate_standalone_watermark_ass(out_ass_path, clip_start, clip_end, source_watermark, account_watermark):
    """Menulis file .ass hanya berisi watermark glass badge (saat mode no_subtitle aktif)."""
    styles_str, dialogues = _build_watermark_ass_section(clip_start, clip_end, source_watermark, account_watermark)
    if not dialogues:
        return
    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_str}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(dialogues) + "\n"

    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)


def generate_ass_subtitles(segments, out_ass_path, clip_start, clip_end, sub_style="tiktok_viral", sub_position=68, watermark_enabled=True, source_watermark="", account_watermark=""):
    """Generate .ass subtitle file for specified style with optional glass-effect watermark badges."""
    _generate_ass_body(
        segments, out_ass_path, clip_start, clip_end, sub_style, sub_position,
        watermark_enabled, source_watermark, account_watermark,
    )

def _generate_ass_body(segments, out_ass_path, clip_start, clip_end, sub_style, sub_position,
                       watermark_enabled, source_watermark, account_watermark):
    has_wm = watermark_enabled and bool((source_watermark or "").strip() or (account_watermark or "").strip())
    extra_styles, extra_dialogues = ("", [])
    if has_wm:
        extra_styles, extra_dialogues = _build_watermark_ass_section(clip_start, clip_end, source_watermark, account_watermark)

    if sub_style in ["none", "no_subtitle"]:
        if has_wm:
            _generate_standalone_watermark_ass(out_ass_path, clip_start, clip_end, source_watermark, account_watermark)
        return

    # Collect words from segments with strict word-level boundary checking
    words = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if seg_end <= clip_start or seg_start >= clip_end:
            continue
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                w_start = float(w.get("start", 0.0))
                w_end = float(w.get("end", 0.0))
                if w_end > clip_start and w_start < clip_end:
                    words.append(w)
        else:
            words.append({"word": seg.get("text", ""), "start": seg_start, "end": seg_end})

    # Sanitize timestamps to guarantee clean, non-overlapping flow
    sanitized_words = _sanitize_word_timestamps(words, clip_start, clip_end)
    if not sanitized_words:
        if has_wm:
            _generate_standalone_watermark_ass(out_ass_path, clip_start, clip_end, source_watermark, account_watermark)
        return

    margin_v = _calc_margin_v(sub_position)

    if sub_style == "minimalist":
        _generate_minimalist_style(sanitized_words, out_ass_path, clip_start, clip_end,
                                   margin_v=margin_v, extra_styles=extra_styles,
                                   extra_dialogues=extra_dialogues)
    elif sub_style == "classic_karaoke":
        _generate_classic_karaoke_style(sanitized_words, out_ass_path, clip_start, clip_end, margin_v=margin_v, extra_styles=extra_styles, extra_dialogues=extra_dialogues)
    elif sub_style == "word_pop":
        _generate_word_pop_style(sanitized_words, out_ass_path, clip_start, clip_end, margin_v=margin_v, extra_styles=extra_styles, extra_dialogues=extra_dialogues)
    else:
        _generate_tiktok_viral_style(sanitized_words, out_ass_path, clip_start, clip_end, margin_v=margin_v, extra_styles=extra_styles, extra_dialogues=extra_dialogues)


def _chunk_words(words, max_words=2, max_chars=14, uppercase=False,
                 break_on_comma=True):
    """Split sanitized words into display chunks that fit the 9:16 safe width.

    The character budget is enforced BEFORE a word is added, so a chunk can
    never exceed ``max_chars`` (except when a single word is itself longer).
    Previously the limit was checked after appending, which produced over-long
    lines that spilled past the left/right safe-zone margins.
    """
    chunks, current_chunk, current_len = [], [], 0
    terminators = ".?!," if break_on_comma else ".?!"

    for word_info in words:
        text = word_info["word"].strip()
        if uppercase:
            text = text.upper()
        if not text:
            continue

        # +1 accounts for the joining space when the chunk already has content.
        text_len = _display_len(text)
        projected = current_len + text_len + (1 if current_chunk else 0)
        if current_chunk and (
            len(current_chunk) >= max_words or projected > max_chars
        ):
            chunks.append(current_chunk)
            current_chunk, current_len = [], 0
            projected = text_len

        current_chunk.append({"word": text, "start": word_info["start"], "end": word_info["end"]})
        current_len = projected

        # A sentence terminator naturally closes the current chunk.
        # Don't break on comma if the chunk only has 1-2 words, to avoid awkward single-word fragments.
        is_hard_stop = text[-1] in ".?!"
        is_comma_break = break_on_comma and text[-1] == "," and len(current_chunk) >= 3
        if is_hard_stop or is_comma_break:
            chunks.append(current_chunk)
            current_chunk, current_len = [], 0

    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def _generate_tiktok_viral_style(words, out_ass_path, clip_start, clip_end, margin_v=600, extra_styles="", extra_dialogues=None):
    """🎬 TikTok / CapCut Viral: Montserrat ExtraBold + Outline Hitam Tajam + Highlight Emerald VisionClipper."""
    styles_block = f"Style: ViralStyle,Montserrat,80,&H00FFFFFF,&H000000FF,&H00000000,&HB0000000,-1,0,0,0,100,100,0.6,0,1,5.5,2.0,2,{SAFE_MARGIN_H},{SAFE_MARGIN_H},{margin_v},1\n"
    if extra_styles:
        styles_block += f"{extra_styles}\n"

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: {SAFE_WRAP_STYLE}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_block}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    limits = STYLE_CHUNK_LIMITS["tiktok_viral"]
    chunks = _chunk_words(words, max_words=limits["max_words"], max_chars=limits["max_chars"], uppercase=False, break_on_comma=False)
    
    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        if extra_dialogues:
            for d in extra_dialogues:
                f.write(f"{d}\n")
        for chunk in chunks:
            for i in range(len(chunk)):
                start_ass = format_ass_timestamp(chunk[i]["start"])
                # Chunk end time matches next word start or chunk end
                next_start = chunk[i+1]["start"] if i + 1 < len(chunk) else chunk[-1]["end"]
                end_ass = format_ass_timestamp(max(chunk[i]["end"], next_start))
                display = []
                for j in range(len(chunk)):
                    raw_word = chunk[j]["word"]
                    if j < i:
                        # Kata yang sudah lewat: putih solid
                        display.append(f"{{\\c&H00FFFFFF&}}{raw_word}")
                    elif j == i:
                        # Kata aktif: Highlight Emerald VisionClipper (#34D399 -> &H0099D334&) dengan pop scale
                        display.append(f"{{\\c&H0099D334&}}{{\\fs88}}{raw_word}{{\\fs80}}{{\\c&H00FFFFFF&}}")
                    else:
                        # Kata mendatang: tetap terlihat dalam putih solid (CapCut style: satu kalimat/frasa utuh terlihat)
                        display.append(f"{{\\c&H00FFFFFF&}}{raw_word}")
                f.write(f"Dialogue: 0,{start_ass},{end_ass},ViralStyle,,0,0,0,,{' '.join(display)}\n")


def _generate_classic_karaoke_style(words, out_ass_path, clip_start, clip_end, margin_v=580, extra_styles="", extra_dialogues=None):
    """🎤 Classic Karaoke: Teks natural case dengan sapuan highlight Electric Cyan / Mint per kata sesuai preview."""
    styles_block = f"Style: KaraokeStyle,Montserrat,84,&H00FFFFFF,&H000000FF,&H00000000,&HB0000000,-1,0,0,0,100,100,0.6,0,1,6.5,2.5,2,{SAFE_MARGIN_H},{SAFE_MARGIN_H},{margin_v},1\n"
    if extra_styles:
        styles_block += f"{extra_styles}\n"

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: {SAFE_WRAP_STYLE}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_block}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # 2-3 words per chunk for readable karaoke phrasing (kept inside the safe width)
    limits = STYLE_CHUNK_LIMITS["classic_karaoke"]
    chunks = _chunk_words(words, max_words=limits["max_words"], max_chars=limits["max_chars"], uppercase=False)
    
    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        if extra_dialogues:
            for d in extra_dialogues:
                f.write(f"{d}\n")
        for chunk in chunks:
            for i in range(len(chunk)):
                start_ass = format_ass_timestamp(chunk[i]["start"])
                next_start = chunk[i+1]["start"] if i + 1 < len(chunk) else chunk[-1]["end"]
                end_ass = format_ass_timestamp(max(chunk[i]["end"], next_start))
                display = []
                for j in range(len(chunk)):
                    raw_word = chunk[j]["word"]
                    if j < i:
                        # Kata yang sudah lewat: Tetap putih bersih
                        display.append(f"{{\\c&H00FFFFFF&}}{raw_word}")
                    elif j == i:
                        # Kata aktif: Highlight Electric Cyan / Mint cerah (&H00D4F500&) dengan pop font
                        display.append(f"{{\\c&H00D4F500&}}{{\\fs94}}{raw_word}{{\\fs84}}{{\\c&H00FFFFFF&}}")
                    else:
                        # Kata mendatang: Putih solid
                        display.append(f"{{\\c&H00FFFFFF&}}{raw_word}")
                f.write(f"Dialogue: 0,{start_ass},{end_ass},KaraokeStyle,,0,0,0,,{' '.join(display)}\n")


def _generate_minimalist_style(words, out_ass_path, clip_start, clip_end, margin_v=560, extra_styles="", extra_dialogues=None):
    """✨ Minimalist: Teks putih bersih statis, elegan dan tidak mengganggu konten podcast."""
    styles_block = f"Style: MinimalStyle,Plus Jakarta Sans,74,&H00FFFFFF,&H000000FF,&H000A0A0A,&HA8000000,-1,0,0,0,100,100,0.4,0,1,4.5,2.0,2,{SAFE_MARGIN_H},{SAFE_MARGIN_H},{margin_v},1\n"
    if extra_styles:
        styles_block += f"{extra_styles}\n"

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: {SAFE_WRAP_STYLE}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_block}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # 3-4 words per line for clean readability (kept inside the safe width)
    limits = STYLE_CHUNK_LIMITS["minimalist"]
    chunks = _chunk_words(words, max_words=limits["max_words"], max_chars=limits["max_chars"], uppercase=False)
    
    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        if extra_dialogues:
            for d in extra_dialogues:
                f.write(f"{d}\n")
        for chunk in chunks:
            start_ass = format_ass_timestamp(chunk[0]["start"])
            end_ass = format_ass_timestamp(chunk[-1]["end"])
            text_line = " ".join(w["word"] for w in chunk)
            f.write(f"Dialogue: 0,{start_ass},{end_ass},MinimalStyle,,0,0,0,,{text_line}\n")


def _generate_word_pop_style(words, out_ass_path, clip_start, clip_end, margin_v=960, extra_styles="", extra_dialogues=None):
    """⚡ Word Pop (Reels): Menampilkan 1 kata besar per saat di posisi tengah."""
    styles_block = f"Style: WordPopStyle,Impact,108,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,0,0,1,12,6,2,{SAFE_MARGIN_H},{SAFE_MARGIN_H},{margin_v},1\n"
    if extra_styles:
        styles_block += f"{extra_styles}\n"

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: {SAFE_WRAP_STYLE}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_block}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Safe usable width and an approximate per-glyph width for Impact (bold).
    usable_w = 1080 - 2 * SAFE_MARGIN_H
    glyph_ratio = 0.60
    base_fs = 108
    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        if extra_dialogues:
            for d in extra_dialogues:
                f.write(f"{d}\n")
        for i, w in enumerate(words):
            start_ass = format_ass_timestamp(w["start"])
            next_start = words[i+1]["start"] if i + 1 < len(words) else w["end"]
            end_ass = format_ass_timestamp(max(w["end"], next_start))
            text = w["word"].upper()
            # Auto-shrink long single words so they never spill past the safe width.
            word_len = max(1, len(w["word"].strip()))
            fit_fs = int(usable_w / (glyph_ratio * word_len))
            fs = max(44, min(base_fs, fit_fs))
            # Dynamic punchy scale effect on entry
            f.write(f"Dialogue: 0,{start_ass},{end_ass},WordPopStyle,,0,0,0,,{{\\fs{fs}}}{{\\t(0,60,\\fscx110\\fscy110)}}{{\\t(60,120,\\fscx100\\fscy100)}}{text}\n")

def detect_speaker_anchors(video_path, start_time, end_time, width, height, target_width, tracking_engine=None):
    """
    Scan the clip to find stable discrete speaker anchors (Left, Right, or Center).
    Prevents the camera from wandering into empty spaces or the center of tables.
    Uses the active tracking engine (YuNet face primary + YOLOv8/Haar fallback) at
    downscaled resolution for fast and reliable multi-speaker detection.
    """
    engine_plan = select_tracking_engine(preferred=tracking_engine)
    primary = engine_plan["primary"]
    fallback = engine_plan["fallback"]
    device = engine_plan["device"]

    ana_w = min(int(width), ANALYSIS_MAX_WIDTH)
    ana_w = max(2, ana_w - (ana_w % 2))
    ana_h = max(2, int(round(height * ana_w / float(width))))
    ana_h = max(2, ana_h - (ana_h % 2))
    scale_x = float(width) / float(ana_w)

    yunet_detector = _get_yunet_detector(ana_w, ana_h, score_threshold=0.42) if "yunet" in (primary, fallback) else None
    yolo_model = _get_yolo_model() if primary == "yolo" else None
    haar_detectors = _get_haar_detectors() if "haar" in (primary, fallback) else None

    MIN_CENTER_X = target_width / 2
    MAX_CENTER_X = width - (target_width / 2)

    def _clamp(cx):
        return max(MIN_CENTER_X, min(cx, MAX_CENTER_X))

    def _run_engine(eng, bgr_small):
        nonlocal yolo_model
        if eng == "yunet" and yunet_detector is not None:
            return _detect_faces_yunet(
                yunet_detector, bgr_small, ana_w, ana_h,
                min_w_ratio=0.04, max_cy_ratio=0.80, min_cy_ratio=0.10,
            )
        if eng == "yolo":
            if yolo_model is None:
                yolo_model = _get_yolo_model()
            if yolo_model is not None:
                return _detect_persons_yolo(
                    yolo_model, bgr_small, ana_w, ana_h, device=device,
                    min_w_ratio=0.06, yunet_detector=yunet_detector,
                )
        if eng == "haar" and haar_detectors:
            return _detect_faces_haar(haar_detectors, bgr_small, ana_w, ana_h, min_w_ratio=0.04)
        return []

    clip_span = max(0.5, float(end_time) - float(start_time))
    num_samples = min(12, max(6, int(round(clip_span / 2.5))))
    sample_times = [round(clip_span * i / float(num_samples), 2) for i in range(num_samples)]
    clip = None
    left_positions = []
    right_positions = []
    solo_positions = []

    try:
        clip = VideoFileClip(video_path).subclipped(start_time, end_time)
        for st in sample_times:
            if st >= clip.duration:
                break
            frame = clip.get_frame(st)
            bgr_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
            if bgr_img.shape[1] != ana_w or bgr_img.shape[0] != ana_h:
                bgr_small = cv2.resize(bgr_img, (ana_w, ana_h), interpolation=cv2.INTER_AREA)
            else:
                bgr_small = bgr_img

            detections = _run_engine(primary, bgr_small)
            if not detections and fallback:
                detections = _run_engine(fallback, bgr_small)

            if detections:
                faces_in_frame = [d["cx"] * scale_x for d in detections]
                if len(faces_in_frame) == 1:
                    solo_positions.append(faces_in_frame[0])
                    if faces_in_frame[0] < width * 0.50:
                        left_positions.append(faces_in_frame[0])
                    else:
                        right_positions.append(faces_in_frame[0])
                elif len(faces_in_frame) >= 2:
                    faces_in_frame.sort()
                    left_positions.append(faces_in_frame[0])
                    right_positions.append(faces_in_frame[-1])
        clip.close()
    except Exception:
        if clip:
            try:
                clip.close()
            except Exception:
                pass

    # Compute median / average anchors
    left_anchor = _clamp(np.median(left_positions)) if left_positions else _clamp(width * 0.25)
    right_anchor = _clamp(np.median(right_positions)) if right_positions else _clamp(width * 0.75)
    solo_anchor = _clamp(np.median(solo_positions)) if solo_positions else left_anchor

    is_multi_speaker = bool(len(left_positions) >= 2 and len(right_positions) >= 2)

    return {
        "left_anchor": float(left_anchor),
        "right_anchor": float(right_anchor),
        "solo_anchor": float(solo_anchor),
        "is_multi_speaker": is_multi_speaker,
        "engine": primary,
    }

def apply_split_screen(video_path, start_time, end_time, output_path, anchors=None, job_id=None):
    """
    Render 9:16 vertical video using Stacked Split-Screen (Top & Bottom).
    Top window: Left speaker / Host (1080x960)
    Bottom window: Right speaker / Guest (1080x960)
    100% static framing, zero panning motion sickness.
    """
    log("Applying Studio Split-Screen (Top: Host / Bottom: Guest)...", job_id=job_id)
    ffmpeg_bin = get_ffmpeg_binary()
    clip_dur = end_time - start_time

    # Determine probe size
    try:
        width, height, _ = _probe_video_stream(video_path)
    except Exception:
        temp_clip = VideoFileClip(video_path)
        width, height = temp_clip.size
        temp_clip.close()

    target_crop_w = int(height * 0.54) # 1:1 roughly or 9:8 ratio for 960h
    if target_crop_w % 2 != 0: target_crop_w -= 1
    
    if anchors is None:
        anchors = detect_speaker_anchors(video_path, start_time, end_time, width, height, target_crop_w)

    left_cx = anchors["left_anchor"]
    right_cx = anchors["right_anchor"]

    # Calculate crop offsets (clamped)
    top_x = max(0, min(int(left_cx - target_crop_w / 2), width - target_crop_w))
    bottom_x = max(0, min(int(right_cx - target_crop_w / 2), width - target_crop_w))

    top_x = (top_x // 2) * 2
    bottom_x = (bottom_x // 2) * 2

    # FFmpeg filter:
    # 1. Split into 2 streams
    # 2. Crop top speaker -> scale to 1080x960
    # 3. Crop bottom speaker -> scale to 1080x960
    # 4. Vertical Stack (vstack) with a sleek subtle border
    fc_filter = (
        f"[0:v]split=2[vtop_in][vbot_in];"
        f"[vtop_in]crop={target_crop_w}:{height}:{top_x}:0,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[top];"
        f"[vbot_in]crop={target_crop_w}:{height}:{bottom_x}:0,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[bot];"
        f"[top][bot]vstack=inputs=2[stacked];"
        f"[stacked]drawbox=x=0:y=958:w=1080:h=4:color=0x18181b@0.9:t=fill[v]"
    )

    cmd = [
        ffmpeg_bin, "-y",
        "-ss", str(start_time),
        "-t", str(clip_dur),
        "-i", video_path,
        "-filter_complex", fc_filter,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]
    run_command(cmd, "Rendering Split-Screen format...", job_id=job_id, duration=clip_dur)
    return output_path

def apply_fit_blur(video_path, start_time, end_time, output_path, job_id=None):
    """
    Render 9:16 vertical video using Fit + Blurred Background.
    Main 16:9 video dinaikkan 5% ke atas (posisi seimbang & proporsional untuk subtitle).
    """
    log("Applying Fit 16:9 with Blurred Background (Video dinaikkan 5% untuk ruang subtitle)...", job_id=job_id)
    ffmpeg_bin = get_ffmpeg_binary()
    clip_dur = end_time - start_time

    # FFmpeg filter:
    # Background: scale to 1080x1920 crop, heavy blur & slight darken
    # Foreground: scale to 1080:-1 (approx 1080x608), dinaikkan 10% dari posisi tengah
    # Posisi tengah = (H-h)/2. Dinaikkan 10% (192px) = (H-h)/2 - (H*0.10)
    fc_filter = (
        "[0:v]split=2[bg_in][fg_in];"
        "[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5,eq=brightness=-0.08[bg];"
        "[fg_in]scale=1080:-1[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2-(H*0.05)[v]"
    )

    cmd = [
        ffmpeg_bin, "-y",
        "-ss", str(start_time),
        "-t", str(clip_dur),
        "-i", video_path,
        "-filter_complex", fc_filter,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]
    run_command(cmd, "Rendering Fit-Blur format...", job_id=job_id, duration=clip_dur)
    return output_path


class MoviePyLogger(ProgressBarLogger):
    def __init__(self, job_id=None):
        super().__init__()
        self.job_id = job_id
        import time
        self.start_time = time.time()
        self.last_log_time = 0

    def bars_callback(self, bar, attr, value, old_value=None):
        import time
        current_time = time.time()
        if current_time - self.last_log_time < 1.0: return
        
        total = self.bars[bar]['total']
        if total > 0:
            percent = (value / total) * 100
            elapsed = current_time - self.start_time
            if percent > 0:
                eta = elapsed / (percent / 100) - elapsed
                from backend.utils import format_timestamp
                eta_str = format_timestamp(eta).split('.')[0]
                log(f"Progress Smart Crop: {percent:.1f}% | Sisa Waktu (ETA): {eta_str}", job_id=self.job_id)
                self.last_log_time = current_time

def _gaussian_smooth(arr, sigma=28):
    """Filter Gaussian 1D untuk menghasilkan pergerakan kamera sinematik/gimbal gliding."""
    radius = int(sigma * 3)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, radius, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


# ═══════════════════════════════════════════════════════════════
# SCENE-CUT DETECTION (podcast multi-kamera)
# ═══════════════════════════════════════════════════════════════
# Skor = rata-rata selisih absolut piksel HSV antar frame (skala 0-255, ala PySceneDetect ContentDetector).
# Lebih andal daripada histogram untuk cut antar angle di studio yang warnanya mirip.
SCENE_CUT_DEFAULT_THRESHOLD = 27.0
SCENE_CUT_MIN_GAP_SEC = 0.4
SCENE_SIGNATURE_WIDTH = 160


def _frame_signature(frame):
    """Thumbnail HSV kecil (int16) untuk perbandingan antar frame yang murah."""
    h, w = frame.shape[:2]
    sig_h = max(2, int(round(h * SCENE_SIGNATURE_WIDTH / float(w))))
    small = cv2.resize(frame, (SCENE_SIGNATURE_WIDTH, sig_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)


class SceneCutDetector:
    """
    Deteksi hard cut secara streaming (kausal), sehingga deteksi wajah bisa dipaksa jalan
    tepat di frame pertama angle baru.
    Menggabungkan skor HSV rata-rata global dan skor grid 4x4 lokal (top-4 blok) agar cut
    antar angle di studio gelap/berwarna seragam tetap tertangkap tanpa sensitif terhadap noise.
    """

    def __init__(self, fps, threshold=SCENE_CUT_DEFAULT_THRESHOLD, min_gap_sec=SCENE_CUT_MIN_GAP_SEC):
        from collections import deque
        self.threshold = float(threshold)
        self.min_gap_frames = max(1, int(round(min_gap_sec * fps)))
        win = max(4, int(round(fps)))
        self.recent_scores = deque(maxlen=win)
        self.recent_block_scores = deque(maxlen=win)
        self.prev_signature = None
        self.last_cut_frame = None
        self.cuts = []

    def update(self, frame_idx, frame):
        signature = _frame_signature(frame)
        prev = self.prev_signature
        self.prev_signature = signature
        if prev is None or prev.shape != signature.shape:
            return False

        diff = np.abs(signature - prev)
        # Hue OpenCV melingkar (0..179): merah/warna kulit di sekitar 0<->179 jangan dianggap beda jauh
        diff[..., 0] = np.minimum(diff[..., 0], 180 - diff[..., 0])
        score = float(np.mean(diff))

        bh, bw = diff.shape[0] // 4, diff.shape[1] // 4
        if bh >= 2 and bw >= 2:
            blocks = diff[: bh * 4, : bw * 4].reshape(4, bh, 4, bw, 3).mean(axis=(1, 3, 4)).reshape(-1)
            top4_score = float(np.mean(np.partition(blocks, -4)[-4:]))
        else:
            top4_score = score

        baseline = float(np.mean(self.recent_scores)) if self.recent_scores else 0.0
        block_baseline = float(np.mean(self.recent_block_scores)) if self.recent_block_scores else 0.0
        self.recent_scores.append(score)
        self.recent_block_scores.append(top4_score)

        is_local_cut = (
            score >= self.threshold * 0.35
            and top4_score >= self.threshold * 0.90
            and top4_score >= 3.8 * max(block_baseline, 2.0)
        )
        is_cut = (
            score >= self.threshold
            or (score >= self.threshold * 0.6 and score >= 3.0 * max(baseline, 1.0))
            or is_local_cut
        )
        if not is_cut:
            return False
        if self.last_cut_frame is not None and frame_idx - self.last_cut_frame < self.min_gap_frames:
            return False  # Flash/strobe: abaikan cut beruntun dalam < 0.4 detik

        self.last_cut_frame = frame_idx
        self.cuts.append(frame_idx)
        # Mulai baseline baru untuk shot baru
        self.recent_scores.clear()
        self.recent_block_scores.clear()
        return True


def _build_shot_trajectory(total_frames, sample_indices, sample_values, cut_frames, deadzone_px, sigma, default_x):
    """
    Bangun posisi center-X per frame, diproses PER SHOT (dipisah oleh cut_frames):
    interpolasi + deadzone + Gaussian smoothing tidak pernah melewati batas cut, sehingga kamera
    langsung "snap" ke angle baru tanpa meluncur.

    sample_values boleh berisi None/NaN (deteksi gagal): di dalam shot posisi ditahan dari deteksi
    sebelumnya; shot tanpa deteksi sama sekali memakai posisi akhir shot sebelumnya (atau default_x).
    """
    total_frames = int(total_frames)
    if total_frames <= 0:
        return np.zeros(0, dtype=np.float64)

    idx = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    vals = np.asarray([np.nan if v is None else v for v in sample_values], dtype=np.float64).reshape(-1)
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]

    bounds = sorted({int(c) for c in cut_frames if 0 < int(c) < total_frames})
    starts = [0] + bounds
    ends = bounds + [total_frames]

    out = np.empty(total_frames, dtype=np.float64)
    prev_final = float(default_x)
    for s, e in zip(starts, ends, strict=True):
        mask = (idx >= s) & (idx < e)
        shot_idx = idx[mask]
        shot_vals = vals[mask].copy()

        # Forward-fill deteksi gagal dengan deteksi terakhir DI DALAM shot ini saja
        last = np.nan
        for k in range(len(shot_vals)):
            if np.isnan(shot_vals[k]):
                shot_vals[k] = last
            else:
                last = shot_vals[k]
        valid = ~np.isnan(shot_vals)

        if not valid.any():
            seg = np.full(e - s, prev_final, dtype=np.float64)
        else:
            # np.interp menahan nilai di tepi -> deteksi pertama shot dipakai mundur ke awal shot
            seg = np.interp(np.arange(s, e), shot_idx[valid], shot_vals[valid])

            # Deadzone Filter (kunci tripod jika gerakan mikro di bawah toleransi), reset per shot
            anchor = seg[0]
            deadzoned = np.empty_like(seg)
            for i, value in enumerate(seg):
                if abs(value - anchor) > deadzone_px:
                    anchor = value
                deadzoned[i] = anchor

            # Gaussian Smoothing (kurva gimbal lembut), hanya di dalam shot
            seg = _gaussian_smooth(deadzoned, sigma=sigma) if len(deadzoned) > 1 else deadzoned

        out[s:e] = seg
        prev_final = float(seg[-1])
    return out


# ═══════════════════════════════════════════════════════════════
# AUDIO: COMPRESSOR + LOUDNORM 2-PASS (target LUFS platform short video)
# ═══════════════════════════════════════════════════════════════
AUDIO_SAMPLE_RATE = 48000
AUDIO_BITRATE = "192k"
DEFAULT_TARGET_LUFS = -14.0   # Standar normalisasi TikTok / Reels / YouTube Shorts
LOUDNORM_TRUE_PEAK = -1.5
LOUDNORM_LRA = 11.0
SILENCE_LUFS_FLOOR = -50.0    # Di bawah ini dianggap (hampir) sunyi -> jangan dinaikkan
_LOUDNORM_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")


def get_audio_settings():
    return {
        "target_lufs": env_float("VC_TARGET_LUFS", DEFAULT_TARGET_LUFS, -30.0, -5.0),
        "denoise": env_bool("VC_AUDIO_DENOISE", False),
    }


def build_audio_pre_chain(denoise=False):
    """Rantai audio sebelum normalisasi: highpass ringan, (opsional) denoise, compressor halus."""
    parts = ["asetpts=PTS-STARTPTS", "highpass=f=80"]
    if denoise:
        parts.append("afftdn=nr=10:nf=-40")
    parts.append("acompressor=threshold=-18dB:ratio=3:attack=5:release=80:makeup=2dB")
    return ",".join(parts)


def _loudnorm_base(target_lufs):
    return f"loudnorm=I={target_lufs:g}:TP={LOUDNORM_TRUE_PEAK:g}:LRA={LOUDNORM_LRA:g}"


def _parse_loudnorm_json(text):
    """Ambil blok JSON hasil `loudnorm=print_format=json` dari stderr FFmpeg. Return dict float atau None."""
    if not text:
        return None
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.S)
    if not blocks:
        return None
    try:
        data = json.loads(blocks[-1])
        return {key: float(data[key]) for key in _LOUDNORM_KEYS}
    except (ValueError, KeyError, TypeError):
        return None


def _measure_loudness(video_path, start, duration, pre_chain, target_lufs, job_id=None):
    """Pass ukur loudnorm (audio saja, tanpa decode video). Return dict hasil ukur atau None jika gagal."""
    cmd = [
        get_ffmpeg_binary(), "-hide_banner", "-nostats",
        "-ss", f"{float(start):.3f}", "-t", f"{float(duration):.3f}",
        "-i", video_path,
        "-vn", "-sn", "-map", "0:a:0",
        "-af", f"{pre_chain},{_loudnorm_base(target_lufs)}:print_format=json",
        "-f", "null", "-",
    ]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=max(60.0, float(duration) * 3), **kwargs)
    except Exception as e:
        log(f"Info: pengukuran loudness gagal ({e}), memakai loudnorm 1-pass.", job_id=job_id)
        return None
    if res.returncode != 0:
        return None
    return _parse_loudnorm_json(res.stderr)


def build_audio_filter(fade_out_start, pre_chain, target_lufs, measurement=None, fade_dur=0.5):
    """
    Return (filter_string, mode). mode:
      - "two_pass": loudnorm linear memakai hasil ukur (paling natural, tanpa pumping)
      - "single_pass": loudnorm dinamis (fallback jika pengukuran gagal)
      - "skip": klip (hampir) sunyi, tidak dinormalisasi agar noise tidak ikut dikeraskan
    """
    loudnorm = _loudnorm_base(target_lufs)
    mode = "single_pass"
    if measurement:
        values = [measurement.get(key) for key in _LOUDNORM_KEYS]
        finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        if not finite or measurement["input_i"] < SILENCE_LUFS_FLOOR:
            loudnorm = None
            mode = "skip"
        else:
            loudnorm += (
                f":measured_I={measurement['input_i']:.2f}"
                f":measured_TP={measurement['input_tp']:.2f}"
                f":measured_LRA={measurement['input_lra']:.2f}"
                f":measured_thresh={measurement['input_thresh']:.2f}"
                f":offset={measurement['target_offset']:.2f}"
                f":linear=true"
            )
            mode = "two_pass"

    parts = [pre_chain]
    if loudnorm:
        parts.append(loudnorm)
    parts.append(f"afade=t=out:st={max(0.0, float(fade_out_start)):.3f}:d={max(0.1, float(fade_dur)):g}")
    return ",".join(parts), mode

_YOLO_MODEL_CACHE = {}
_YUNET_DETECTOR_CACHE = {}
_HAAR_CASCADE_CACHE = {}
_YUNET_LOCK = threading.Lock()

VALID_TRACKING_ENGINES = ("auto", "yunet", "yolo", "haar")

# Lebar maksimum frame untuk Pass 1 (deteksi). 640px (360p untuk 16:9) adalah
# sweet-spot untuk YuNet (dilatih pada 320x320) maupun YOLOv8 (imgsz=384):
# memangkas bandwidth pipe FFmpeg & waktu inferensi YuNet >50% dibanding 960p.
ANALYSIS_MAX_WIDTH = 640

# Fallback statis jika tracking gagal: kunci kursi pembicara (kiri ~28%), BUKAN tengah meja (50%).
STATIC_FALLBACK_CROP_CHAIN = (
    "setpts=PTS-STARTPTS,crop=ih*9/16:ih:iw*0.14:0,"
    "scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos+accurate_rnd+full_chroma_int,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
)


def _is_yunet_available():
    """Return True jika OpenCV FaceDetectorYN dan model ONNX YuNet tersedia."""
    try:
        return bool(
            hasattr(cv2, "FaceDetectorYN")
            and os.path.isfile(YUNET_MODEL_PATH)
            and os.path.getsize(YUNET_MODEL_PATH) > 0
        )
    except OSError:
        return False


def _is_yolo_available():
    """Return True jika paket ultralytics dan bobot yolov8n.pt tersedia (tanpa import berat)."""
    try:
        if not (os.path.isfile(YOLO_MODEL_PATH) and os.path.getsize(YOLO_MODEL_PATH) > 0):
            return False
        if "ultralytics" in sys.modules:
            return sys.modules["ultralytics"] is not None
        import importlib.util
        return importlib.util.find_spec("ultralytics") is not None
    except Exception:
        return False


def _is_haar_available():
    """Return True jika OpenCV CascadeClassifier dan file XML Haar tersedia."""
    try:
        return bool(
            hasattr(cv2, "CascadeClassifier")
            and os.path.isfile(CASCADE_PATH)
            and os.path.getsize(CASCADE_PATH) > 0
        )
    except OSError:
        return False


def _get_torch_device():
    try:
        # Hindari import torch yang lambat di Linux CPU-only jika tidak ada device NVIDIA
        if (
            "torch" not in sys.modules
            and sys.platform.startswith("linux")
            and not os.path.exists("/dev/nvidia0")
            and not os.path.exists("/proc/driver/nvidia")
        ):
            return "cpu"
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def detect_tracking_engines():
    """
    Deteksi engine face/person tracking yang tersedia di sistem saat ini
    (YuNet ONNX, YOLOv8, Haar Cascade) beserta akselerator perangkat kerasnya.
    """
    yunet_ok = _is_yunet_available()
    yolo_ok = _is_yolo_available()
    haar_ok = _is_haar_available()
    device = _get_torch_device()

    available = []
    if yunet_ok:
        available.append("yunet")
    if yolo_ok:
        available.append("yolo")
    if haar_ok:
        available.append("haar")

    # YuNet diutamakan sebagai primary karena mendeteksi titik tengah WAJAH secara
    # langsung dan berjalan jauh lebih ringan/cepat di OpenCV C++ DNN (~8-12ms/frame),
    # sedangkan YOLOv8 menjadi fallback otomatis saat wajah terhalang/menoleh jauh.
    if yunet_ok:
        recommended = "yunet"
        fallback = "yolo" if yolo_ok else ("haar" if haar_ok else None)
    elif yolo_ok:
        recommended = "yolo"
        fallback = "haar" if haar_ok else None
    elif haar_ok:
        recommended = "haar"
        fallback = None
    else:
        recommended = "none"
        fallback = None

    raw_cfg = (
        os.environ.get("VC_TRACKING_ENGINE")
        or os.environ.get("VC_FACE_ENGINE")
        or "auto"
    ).strip().lower()

    return {
        "yunet": yunet_ok,
        "yolo": yolo_ok,
        "haar": haar_ok,
        "device": device,
        "available": available,
        "recommended": recommended,
        "fallback": fallback,
        "configured": raw_cfg,
    }


def select_tracking_engine(preferred=None, job_id=None):
    """
    Pilih engine tracking (primary & fallback) paling efisien berdasarkan engine
    yang terdeteksi di sistem dan preferensi pengguna (`auto`, `yunet`, `yolo`, `haar`).
    """
    info = detect_tracking_engines()
    req = (preferred or info["configured"] or "auto").strip().lower()
    if req in ("yolov8", "ultralytics"):
        req = "yolo"
    if req not in VALID_TRACKING_ENGINES:
        log(f"Warning: VC_TRACKING_ENGINE='{req}' tidak dikenal, kembali ke 'auto'.", job_id=job_id)
        req = "auto"

    primary = info["recommended"]
    fallback = info["fallback"]

    if req == "yunet":
        if info["yunet"]:
            primary = "yunet"
            fallback = "haar" if info["haar"] else None
        else:
            log(f"Warning: Engine 'yunet' diminta tetapi tidak tersedia, beralih ke '{primary}'.", job_id=job_id)
    elif req == "yolo":
        if info["yolo"]:
            primary = "yolo"
            fallback = "yunet" if info["yunet"] else ("haar" if info["haar"] else None)
        else:
            log(f"Warning: Engine 'yolo' diminta tetapi tidak tersedia, beralih ke '{primary}'.", job_id=job_id)
    elif req == "haar":
        if info["haar"]:
            primary = "haar"
            fallback = None
        else:
            log(f"Warning: Engine 'haar' diminta tetapi tidak tersedia, beralih ke '{primary}'.", job_id=job_id)

    return {
        "requested": req,
        "primary": primary,
        "fallback": fallback,
        "device": info["device"],
        "available": info["available"],
    }


def _get_yunet_detector(width, height, score_threshold=0.55, nms_threshold=0.3):
    """Ambil atau buat instance OpenCV FaceDetectorYN (YuNet) dengan ukuran input yang sesuai."""
    if not _is_yunet_available():
        return None
    w, h = int(width), int(height)
    key = (round(float(score_threshold), 2), round(float(nms_threshold), 2))
    with _YUNET_LOCK:
        det = _YUNET_DETECTOR_CACHE.get(key)
        if det is None:
            try:
                det = cv2.FaceDetectorYN.create(
                    YUNET_MODEL_PATH, "", (w, h),
                    score_threshold=float(score_threshold),
                    nms_threshold=float(nms_threshold),
                )
                _YUNET_DETECTOR_CACHE[key] = det
            except Exception:
                return None
        else:
            try:
                det.setInputSize((w, h))
            except Exception:
                try:
                    det = cv2.FaceDetectorYN.create(
                        YUNET_MODEL_PATH, "", (w, h),
                        score_threshold=float(score_threshold),
                        nms_threshold=float(nms_threshold),
                    )
                    _YUNET_DETECTOR_CACHE[key] = det
                except Exception:
                    return None
        return det


def _get_yolo_model(job_id=None):
    """Load YOLOv8 sekali per proses (lazy) lalu dipakai ulang untuk semua klip."""
    if "model" not in _YOLO_MODEL_CACHE:
        try:
            from ultralytics import YOLO
            _YOLO_MODEL_CACHE["model"] = YOLO(YOLO_MODEL_PATH)
        except Exception as e:
            log(f"Warning: YOLOv8 failed to load ({e}), falling back to YuNet Face Detector...", job_id=job_id)
            _YOLO_MODEL_CACHE["model"] = None
    return _YOLO_MODEL_CACHE["model"]


def _get_haar_detectors():
    """Load Haar Cascade frontal & profile sebagai fallback ringan terakhir."""
    if "detectors" not in _HAAR_CASCADE_CACHE:
        detectors = []
        if _is_haar_available():
            for path in (CASCADE_PATH, PROFILE_CASCADE_PATH):
                if os.path.isfile(path):
                    try:
                        c = cv2.CascadeClassifier(path)
                        if not c.empty():
                            detectors.append(c)
                    except Exception:
                        pass
        _HAAR_CASCADE_CACHE["detectors"] = detectors
    return _HAAR_CASCADE_CACHE["detectors"]


def _extract_normalized_patch(frame, cx, cy, pw, ph, out_size=(24, 16)):
    """Ambil patch grayscale ter-normalisasi (zero-mean) untuk mengukur gerak bibir/dahi."""
    if frame is None or pw < 4 or ph < 4:
        return None
    ih, iw = frame.shape[:2]
    x1 = max(0, int(round(cx - pw / 2.0)))
    y1 = max(0, int(round(cy - ph / 2.0)))
    x2 = min(iw, int(round(cx + pw / 2.0)))
    y2 = min(ih, int(round(cy + ph / 2.0)))
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    roi = frame[y1:y2, x1:x2]
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(roi, out_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return resized - float(np.mean(resized))


def _is_valid_yunet_landmarks(f, fx, fy, fw, fh):
    """
    Validasi anatomi 5 titik landmark YuNet (mata kanan, mata kiri, hidung, sudut mulut)
    untuk membuang deteksi palsu (pareidolia) pada mikrofon, gelas, kepalan tangan, atau logo,
    sambil tetap menerima wajah frontal, 3/4, maupun profil samping.
    """
    if len(f) < 14:
        return True
    pts = [float(f[i]) for i in range(4, 14)]
    if not all(math.isfinite(v) for v in pts):
        return False
    rex, rey, lex, ley, nx, ny, rmx, rmy, lmx, lmy = pts

    pad_x = fw * 0.22
    pad_y = fh * 0.22
    for px, py in ((rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy)):
        if not (fx - pad_x <= px <= fx + fw + pad_x and fy - pad_y <= py <= fy + fh + pad_y):
            return False

    eye_cy = 0.5 * (rey + ley)
    mouth_cy = 0.5 * (rmy + lmy)
    # Mata wajib di atas mulut secara proporsional (mendukung kemiringan kepala wajar)
    if (mouth_cy - eye_cy) < fh * 0.14:
        return False
    if eye_cy > fy + fh * 0.68 or mouth_cy < fy + fh * 0.40:
        return False
    # Hidung berada di pita vertikal antara mata dan mulut
    if ny < min(rey, ley) - fh * 0.10 or ny > max(rmy, lmy) + fh * 0.08:
        return False
    # Kemiringan mata tidak ekstrem vertikal & mata kanan tidak terbalik melewati mata kiri
    if abs(rey - ley) > fh * 0.42:
        return False
    if (rex - lex) > fw * 0.22:
        return False
    return True


def _detect_faces_yunet(
    yunet_detector, frame, ana_w, ana_h,
    min_w_ratio=0.035, max_cy_ratio=0.80, min_cy_ratio=0.05, min_score=None,
):
    """Deteksi wajah dengan YuNet + validasi landmark anatomis + patch mulut/dahi."""
    if yunet_detector is None:
        return []
    try:
        with _YUNET_LOCK:
            yunet_detector.setInputSize((int(ana_w), int(ana_h)))
            _, raw_faces = yunet_detector.detect(frame)
    except Exception:
        return []
    if raw_faces is None or len(raw_faces) == 0:
        return []

    valid = []
    for f in raw_faces:
        fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        score = float(f[-1]) if len(f) > 4 else 1.0
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0
        if fw < ana_w * min_w_ratio or fh < ana_h * 0.04:
            continue
        if cx < ana_w * 0.03 or cx > ana_w * 0.97:
            continue
        if cy < ana_h * min_cy_ratio or cy > ana_h * max_cy_ratio:
            continue
        aspect = fh / max(fw, 1.0)
        if aspect < 0.75 or aspect > 2.25:
            continue

        if min_score is not None:
            req_score = float(min_score)
        elif fw < ana_w * 0.055:
            req_score = 0.70  # Wajah kecil di latar wajib skor tinggi agar bebas pareidolia objek
        elif fw < ana_w * 0.075:
            req_score = 0.60
        else:
            req_score = 0.55
        if score < req_score:
            continue

        if not _is_valid_yunet_landmarks(f, fx, fy, fw, fh):
            continue

        # Landmark mulut YuNet (index 10..13): gunakan jika valid di dalam box wajah
        if len(f) >= 14:
            rmx, rmy, lmx, lmy = float(f[10]), float(f[11]), float(f[12]), float(f[13])
            mcx = (rmx + lmx) / 2.0
            mcy = (rmy + lmy) / 2.0
            if not (fx <= mcx <= fx + fw and fy <= mcy <= fy + fh):
                mcx, mcy = cx, fy + fh * 0.72
        else:
            mcx, mcy = cx, fy + fh * 0.72

        mouth_patch = _extract_normalized_patch(frame, mcx, mcy, fw * 0.48, fh * 0.28, out_size=(24, 16))
        forehead_patch = _extract_normalized_patch(frame, cx, fy + fh * 0.24, fw * 0.48, fh * 0.20, out_size=(24, 12))
        is_corner_pip = bool(
            fw < ana_w * 0.068
            and (cx < ana_w * 0.20 or cx > ana_w * 0.80)
            and (cy < ana_h * 0.28 or cy > ana_h * 0.72)
        )

        valid.append({
            "cx": cx,
            "cy": cy,
            "w": fw,
            "h": fh,
            "area": fw * fh,
            "score": score,
            "mouth_patch": mouth_patch,
            "forehead_patch": forehead_patch,
            "is_corner_pip": is_corner_pip,
            "engine": "yunet",
        })
    valid.sort(key=lambda d: d["area"], reverse=True)
    if len(valid) > 1 and valid[0]["score"] >= 0.72 and valid[0]["w"] >= ana_w * 0.06:
        max_area = valid[0]["area"]
        valid = [
            d for d in valid
            if not (d["score"] < 0.68 and d["area"] < max_area * 0.45)
        ]
    return valid


def _refine_person_head_with_yunet(yunet_detector, frame, bx1, by1, bx2, by2):
    """
    Saat YOLOv8 mendeteksi kotak 'person' (yang sering melebar ke lengan/meja),
    crop bagian atas (kepala/bahu ~46% tinggi box), normalisasi resolusi crop, dan
    jalankan YuNet sensitif (score >= 0.38) agar titik X mengunci WAJAH, bukan badan/meja.
    """
    if yunet_detector is None or frame is None:
        return None
    fh_img, fw_img = frame.shape[:2]
    x1 = max(0, int(math.floor(bx1)))
    y1 = max(0, int(math.floor(by1)))
    x2 = min(fw_img, int(math.ceil(bx2)))
    y2_head = min(fh_img, int(math.ceil(by1 + (by2 - by1) * 0.46)))
    cw, ch = x2 - x1, y2_head - y1
    if cw < 24 or ch < 24:
        return None
    head_crop = frame[y1:y2_head, x1:x2]
    scale = 1.0
    if cw < 180:
        scale = min(2.5, 180.0 / float(cw))
    elif cw > 340:
        scale = 340.0 / float(cw)
    if abs(scale - 1.0) > 0.05:
        head_crop = cv2.resize(
            head_crop,
            (max(2, int(round(cw * scale))), max(2, int(round(ch * scale)))),
            interpolation=cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA,
        )
    ch_s, cw_s = head_crop.shape[:2]
    head_det = _get_yunet_detector(cw_s, ch_s, score_threshold=0.38) or yunet_detector
    faces = _detect_faces_yunet(
        head_det, head_crop, cw_s, ch_s,
        min_w_ratio=0.08, max_cy_ratio=0.95, min_cy_ratio=0.02, min_score=0.38,
    )
    if not faces:
        return None
    best = faces[0]
    return {
        "cx": x1 + best["cx"] / scale,
        "cy": y1 + best["cy"] / scale,
        "w": best["w"] / scale,
        "h": best["h"] / scale,
    }


def _detect_persons_yolo(yolo_model, frame, ana_w, ana_h, device="cpu", min_w_ratio=0.05, yunet_detector=None):
    """
    Deteksi narasumber (person class=0) dengan YOLOv8; kembalikan list dict terurut
    dari area terbesar. Menolak objek bawah meja/kursi kosong/gabungan 2 orang (>72% frame),
    tetap mendukung solo close-up lebar (50-72%), dan mengunci kepala via YuNet.
    """
    if yolo_model is None:
        return []
    try:
        results = yolo_model.predict(
            source=frame,
            classes=[0],
            conf=0.50,
            imgsz=384,
            device=device,
            verbose=False,
        )
    except Exception:
        return []
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    try:
        boxes = results[0].boxes.xyxy.cpu().numpy()
    except Exception:
        return []

    confs = None
    try:
        raw_confs = results[0].boxes.conf.cpu().numpy()
        if isinstance(raw_confs, np.ndarray) and len(raw_confs) == len(boxes):
            confs = raw_confs
    except Exception:
        confs = None

    valid = []
    max_w_ratio = 0.72
    for idx_b, box in enumerate(boxes):
        bx1, by1, bx2, by2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        conf = float(confs[idx_b]) if confs is not None else 1.0
        bw = bx2 - bx1
        bh = by2 - by1
        if bw <= ana_w * min_w_ratio or bh <= ana_h * 0.25:
            continue
        # Puncak kepala manusia nyata berada di paruh atas frame (bukan kaki di bawah meja/tangan)
        if by1 > ana_h * 0.58:
            continue
        if bh < bw * 0.52:
            continue
        if bw > ana_w * max_w_ratio:
            continue

        head = _refine_person_head_with_yunet(yunet_detector, frame, bx1, by1, bx2, by2)
        if bw > ana_w * 0.50 and head is None:
            # Box lebar (50-72%) tanpa konfirmasi kepala tunggal hanya diterima jika 1 subjek close-up sangat yakin
            if not (len(boxes) == 1 and conf >= 0.72):
                continue
        if head is None and confs is not None and conf < 0.60:
            # Hindari kursi/stand mikrofon skor tanggung tanpa wajah lolos sebagai pembicara
            continue

        cx = head["cx"] if head is not None else (bx1 + bw / 2.0)
        cy = head["cy"] if head is not None else (by1 + bh * 0.24)
        valid.append({
            "cx": cx,
            "cy": cy,
            "w": bw,
            "h": bh,
            "area": bw * bh,
            "score": conf,
            "has_head": head is not None,
            "engine": "yolo",
        })
    valid.sort(key=lambda d: d["area"], reverse=True)
    if len(valid) > 1:
        max_area = valid[0]["area"]
        valid = [d for d in valid if d["has_head"] or d["area"] >= max_area * 0.35]
    return valid


def _detect_faces_haar(haar_detectors, frame, ana_w, ana_h, min_w_ratio=0.04):
    """Fallback deteksi wajah memakai OpenCV Haar Cascade jika YuNet & YOLO tidak tersedia."""
    if not haar_detectors:
        return []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        min_w = max(16, int(ana_w * min_w_ratio))
        valid = []
        for cascade in haar_detectors:
            rects = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_w, min_w))
            if rects is not None and len(rects) > 0:
                for (x, y, w, h) in rects:
                    fw, fh = float(w), float(h)
                    cx = float(x) + fw / 2.0
                    cy = float(y) + fh / 2.0
                    if cy > ana_h * 0.85:
                        continue
                    valid.append({
                        "cx": cx,
                        "cy": cy,
                        "w": fw,
                        "h": fh,
                        "area": fw * fh,
                        "score": 1.0,
                        "engine": "haar",
                    })
                if valid:
                    break
        valid.sort(key=lambda d: d["area"], reverse=True)
        return valid
    except Exception:
        return []


class _MouthActivityTracker:
    """
    Hitung aktivitas gerak bibir/mulut dan liveness kinematik per pembicara.
    Mengurangi gerak dahi/kepala kaku (rigid head motion) agar anggukan pendengar
    tidak dianggap bicara, sekaligus menandai objek wajah statis mati (poster/foto/patung
    di latar) agar tidak mencuri fokus saat pembicara asli menoleh.
    """

    def __init__(self, ana_w, ema_alpha=0.35):
        self.ana_w = float(ana_w)
        self.ema_alpha = float(ema_alpha)
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, detections):
        if not detections:
            return detections
        match_tol = self.ana_w * 0.15
        new_tracks = []
        for d in detections:
            cx = float(d["cx"])
            cy = float(d.get("cy", 0.0))
            fw = float(d.get("w", 0.0))
            mouth = d.get("mouth_patch")
            forehead = d.get("forehead_patch")
            best_prev = None
            best_dist = match_tol
            for tr in self.tracks:
                dist = abs(tr["cx"] - cx)
                if dist <= best_dist:
                    best_dist = dist
                    best_prev = tr

            ema = best_prev["ema"] if best_prev is not None else 0.0
            motion_ema = best_prev["motion_ema"] if best_prev is not None else 0.0
            peak_speech = best_prev["peak_speech"] if best_prev is not None else 0.0
            age = (best_prev["age"] + 1) if best_prev is not None else 1

            mouth_diff = 0.0
            forehead_diff = 0.0
            if (
                best_prev is not None
                and mouth is not None
                and best_prev["mouth"] is not None
                and mouth.shape == best_prev["mouth"].shape
            ):
                mouth_diff = float(np.mean(np.abs(mouth - best_prev["mouth"])))
                if (
                    forehead is not None
                    and best_prev["forehead"] is not None
                    and forehead.shape == best_prev["forehead"].shape
                ):
                    forehead_diff = float(np.mean(np.abs(forehead - best_prev["forehead"])))
                instant = max(0.0, mouth_diff - 0.7 * forehead_diff)
                ema = (1.0 - self.ema_alpha) * ema + self.ema_alpha * instant

            if best_prev is not None:
                pos_jitter = math.hypot(cx - best_prev["cx"], cy - best_prev["cy"]) + 0.5 * abs(fw - best_prev["w"])
                liveness_inst = pos_jitter + 0.25 * (mouth_diff + forehead_diff)
                motion_ema = 0.65 * motion_ema + 0.35 * liveness_inst

            peak_speech = max(peak_speech, ema)
            d["speech_score"] = ema
            d["is_static_prop"] = bool(
                mouth is not None and age >= 5 and peak_speech < 0.85 and motion_ema < 0.35
            )
            new_tracks.append({
                "cx": cx,
                "cy": cy,
                "w": fw,
                "mouth": mouth,
                "forehead": forehead,
                "ema": ema,
                "motion_ema": motion_ema,
                "peak_speech": peak_speech,
                "age": age,
            })
        self.tracks = new_tracks
        return detections


class _SpeakerTargetSelector:
    """
    Pilih satu target pembicara yang stabil per shot pada video split-screen / dua orang.
    Mengutamakan pembicara yang mulutnya aktif bergerak (speaking mouth detector),
    menolak objek statis latar (poster/foto), dan mencegah kamera ping-pong ke garis tengah.
    """

    def __init__(self, ana_w, switch_streak=6, dominance_ratio=1.55, speech_switch_streak=5, min_hold_samples=5):
        self.ana_w = float(ana_w)
        self.switch_streak = int(switch_streak)
        self.dominance_ratio = float(dominance_ratio)
        self.speech_switch_streak = int(speech_switch_streak)
        self.min_hold_samples = int(min_hold_samples)
        self.locked_cx = None
        self.locked_area = 0.0
        self.challenger_streak = 0
        self.samples_since_lock = 0
        self.last_switch_delay = 0

    def reset(self):
        self.locked_cx = None
        self.locked_area = 0.0
        self.challenger_streak = 0
        self.samples_since_lock = 0
        self.last_switch_delay = 0

    def select(self, detections):
        """Return (chosen_detection, switched_far) dari list deteksi."""
        self.last_switch_delay = 0
        if not detections:
            self.samples_since_lock += 1
            return None, False

        if self.locked_cx is None:
            best = max(
                detections,
                key=lambda d: (
                    not bool(d.get("is_static_prop", False)),
                    float(d.get("speech_score", 0.0)) > 2.5,
                    float(d.get("area", 0.0)),
                ),
            )
            self.locked_cx = best["cx"]
            self.locked_area = float(best.get("area", 0.0))
            self.challenger_streak = 0
            self.samples_since_lock = 1
            return best, False

        self.samples_since_lock += 1
        near_tol = self.ana_w * 0.22
        near = [d for d in detections if abs(d["cx"] - self.locked_cx) <= near_tol]
        distant = [
            d for d in detections
            if abs(d["cx"] - self.locked_cx) > self.ana_w * 0.25
            and not (d.get("is_static_prop") and float(d.get("area", 0.0)) < self.locked_area * 0.85)
        ]
        top = detections[0]

        if near:
            anchor_det = max(near, key=lambda d: float(d.get("area", 0.0)))
            anchor_area = float(anchor_det.get("area", 0.0))
            anchor_speech = float(anchor_det.get("speech_score", 0.0))

            best_speaker = max(
                distant,
                key=lambda d: (float(d.get("speech_score", 0.0)), float(d.get("area", 0.0))),
            ) if distant else None

            is_speaking_challenger = (
                best_speaker is not None
                and anchor_speech < 7.5
                and float(best_speaker.get("speech_score", 0.0)) > max(anchor_speech * 1.45, 3.8)
            )
            is_area_challenger = (
                not top.get("is_static_prop")
                and abs(top["cx"] - self.locked_cx) > self.ana_w * 0.25
                and float(top.get("area", 0.0)) >= max(anchor_area, self.locked_area, 1.0) * self.dominance_ratio
            )

            challenger = best_speaker if is_speaking_challenger else (top if is_area_challenger else None)
            needed_streak = self.speech_switch_streak if is_speaking_challenger else self.switch_streak

            if challenger is not None:
                self.challenger_streak += 1
                if self.challenger_streak >= needed_streak and self.samples_since_lock >= self.min_hold_samples:
                    switched_far = abs(challenger["cx"] - self.locked_cx) > self.ana_w * 0.28
                    self.last_switch_delay = needed_streak
                    self.locked_cx = challenger["cx"]
                    self.locked_area = float(challenger.get("area", 0.0))
                    self.challenger_streak = 0
                    self.samples_since_lock = 1
                    return challenger, switched_far
            else:
                # Decay lembut agar jeda nafas 1 sampel di sela kalimat tidak membatalkan switch
                self.challenger_streak = max(0, self.challenger_streak - 1)

            self.locked_cx = anchor_det["cx"]
            self.locked_area = anchor_area
            return anchor_det, False

        # Pembicara yang terkunci tidak terlihat di frame ini (mis. menoleh/minum):
        # tolak kandidat objek statis kecil (poster/patung dinding) agar kamera diam di kursi pembicara.
        live_candidates = [
            d for d in detections
            if not (d.get("is_static_prop") and float(d.get("area", 0.0)) < self.locked_area * 0.80)
        ]
        if not live_candidates:
            return None, False

        cand = live_candidates[0]
        miss_needed = self.switch_streak if float(cand.get("speech_score", 0.0)) >= 3.5 else (self.switch_streak + 2)
        self.challenger_streak += 1
        if self.challenger_streak >= miss_needed:
            switched_far = abs(cand["cx"] - self.locked_cx) > self.ana_w * 0.28
            self.last_switch_delay = miss_needed
            self.locked_cx = cand["cx"]
            self.locked_area = float(cand.get("area", 0.0))
            self.challenger_streak = 0
            self.samples_since_lock = 1
            return cand, switched_far

        # Tahan posisi (return None -> forward-fill dalam shot) selama masih di bawah threshold switch
        return None, False


def _probe_video_stream(video_path):
    """Return (width, height, fps) of the first video stream using ffprobe (supports H.264/VP9/AV1)."""
    ffprobe_bin = get_ffprobe_binary()
    cmd = [
        ffprobe_bin, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of", "json", video_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe gagal membaca video: {res.stderr.strip()[:200]}")
    streams = json.loads(res.stdout or "{}").get("streams", [])
    if not streams:
        raise RuntimeError(f"Tidak ada stream video di: {video_path}")

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Resolusi video tidak valid: {width}x{height}")

    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(stream.get(key) or "0/0")
        try:
            num, den = raw.split("/") if "/" in raw else (raw, "1")
            if float(num) > 0 and float(den) > 0:
                fps = float(num) / float(den)
                break
        except ValueError:
            continue
    if fps <= 0 or fps > 240:
        fps = 30.0
    return width, height, fps


def _iter_ffmpeg_frames(video_path, start_time, duration, fps, out_w, out_h):
    """
    Decode segmen video lewat FFmpeg (seek akurat + CFR) dan yield frame BGR berukuran out_w x out_h.
    Memakai decoder yang sama dengan render final sehingga trajektori crop sinkron dengan audio.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin, "-v", "error",
        "-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}",
        "-i", video_path,
        "-an", "-sn",
        "-vf", f"setpts=PTS-STARTPTS,fps={fps:.6f},scale={out_w}:{out_h}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000
    frame_bytes = out_w * out_h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 2, **kwargs)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()


_TRAJECTORY_CACHE_LOCK = threading.Lock()

TRAJECTORY_ALGO_VERSION = "v10"
FIT_BLUR_ZOOM_DEFAULT = 1.78


def _is_sample_wide_or_no_face(meta, ana_w):
    """Return True jika sampel ini adalah shot wide/zoom-out, screen-share PiP, atau tidak ada wajah jelas."""
    if meta["count"] == 0 or meta.get("is_corner_pip", False):
        return True
    if not meta["is_face"]:
        # Jika YOLO mendeteksi 1 orang besar close-up (>= 22% lebar frame) saat menoleh sesaat,
        # tetap smart_crop; selain itu (wide/multi-person tanpa wajah) -> fit_blur.
        return not (meta["count"] == 1 and meta["max_w"] >= float(ana_w) * 0.22)
    face_w_ratio = meta["max_w"] / float(ana_w)
    min_w_ratio = float(meta.get("min_w") if meta.get("min_w") is not None else meta["max_w"]) / float(ana_w)
    if face_w_ratio < 0.070:
        return True
    if (
        meta["count"] >= 2
        and meta["min_cx"] is not None
        and (meta["max_cx"] - meta["min_cx"]) > ana_w * 0.35
        and (face_w_ratio < 0.085 or min_w_ratio < 0.060)
    ):
        return True
    return False


def _suppress_short_burst_spikes(sample_indices, sample_positions, cut_frames, orig_w, max_burst=3):
    """
    Buang lonjakan posisi sesaat (1..max_burst sampel, ~0.16-0.5s) di dalam satu shot
    akibat distraksi tangan, gelas, mikrofon, atau glitch deteksi singkat yang segera
    kembali ke posisi pembicara semula.
    """
    n = len(sample_positions)
    if n < 3:
        return
    spike_tol = orig_w * 0.22
    neighbor_tol = orig_w * 0.14

    def _shot_start_for(f_idx):
        return max((c for c in cut_frames if c <= f_idx), default=0)

    k = 1
    while k < n - 1:
        prev_p = sample_positions[k - 1]
        if prev_p is None:
            k += 1
            continue
        matched = False
        shot_prev = _shot_start_for(sample_indices[k - 1])
        for b in range(min(max_burst, n - 1 - k), 0, -1):
            after_k = k + b
            next_p = sample_positions[after_k]
            if next_p is None:
                continue
            if _shot_start_for(sample_indices[after_k]) != shot_prev:
                continue
            if abs(prev_p - next_p) > neighbor_tol:
                continue
            burst_vals = sample_positions[k:after_k]
            if all(
                v is not None and abs(v - prev_p) > spike_tol and abs(v - next_p) > spike_tol
                for v in burst_vals
            ):
                for m in range(k, after_k):
                    sample_positions[m] = None
                k = after_k
                matched = True
                break
        if not matched:
            k += 1


def _compute_adaptive_fit_blur_plan(shot_modes, sample_indices, sample_meta, sample_positions, ana_w, orig_w):
    """
    Hitung zoom Fit-Blur yang adaptif dengan isi video asli + sesuaikan titik tengah segmen wide:
      - B-roll tanpa wajah / slide / screen-share dengan facecam pojok -> zoom 1.0x (100% utuh).
      - Wide 2+ pembicara -> zoom dinamis (1.0x - 1.52x) agar KEDUA pembicara + bahu masuk utuh,
        di-center pada rentang kedua pembicara sehingga tidak memotong kepala atau menyorot meja kosong.
      - Wide 1 pembicara jauh -> zoom 1.50x dikunci ke posisi X pembicara tersebut.
    """
    fit_segments = [seg for seg in (shot_modes or []) if seg.get("mode") == "fit_blur"]
    if not fit_segments:
        return 1.0

    scale_x = float(orig_w) / float(max(1, ana_w))
    seg_zooms = []
    for seg in fit_segments:
        s_f, e_f = int(seg.get("start", 0)), int(seg.get("end", 0))
        seg_k = [k for k, idx in enumerate(sample_indices) if s_f <= idx < e_f]
        if not seg_k:
            seg_zooms.append(1.0)
            continue

        empty_or_pip = sum(
            1 for k in seg_k
            if sample_meta[k]["count"] == 0 or sample_meta[k].get("is_corner_pip", False)
        )
        multi_k = [
            k for k in seg_k
            if sample_meta[k]["count"] >= 2
            and sample_meta[k]["min_cx"] is not None
            and sample_meta[k]["max_cx"] is not None
        ]
        if empty_or_pip * 2 >= len(seg_k):
            seg_zooms.append(1.0)
        elif len(multi_k) >= max(1, int(len(seg_k) * 0.30)):
            med_min_cx = float(np.median([sample_meta[k]["min_cx"] for k in multi_k]))
            med_max_cx = float(np.median([sample_meta[k]["max_cx"] for k in multi_k]))
            med_max_w = float(np.median([sample_meta[k]["max_w"] for k in multi_k]))
            pad = max(med_max_w * 2.4, float(ana_w) * 0.13)
            left_edge = max(0.0, med_min_cx - pad)
            right_edge = min(float(ana_w), med_max_cx + pad)
            span_ratio = max(0.66, min(1.0, (right_edge - left_edge) / float(ana_w)))
            z = round(min(1.52, max(1.0, 1.0 / span_ratio)), 2)
            seg_zooms.append(z)
            pair_cx_orig = 0.5 * (left_edge + right_edge) * scale_x
            for k in seg_k:
                sample_positions[k] = pair_cx_orig
        else:
            seg_zooms.append(1.50)

    return float(min(seg_zooms)) if seg_zooms else 1.0


def _classify_shot_modes(total_frames, cut_frames, sample_indices, sample_meta, ana_w):
    """
    Klasifikasikan segmen video menjadi 'smart_crop' vs 'fit_blur' (zoom 1.78x):
      - Bekerja PER SHOT maupun PER RUN SAMPEL (hysteresis), sehingga saat video
        melakukan zoom-out halus atau cut antar angle dengan warna studio mirip
        (tanpa terdeteksi scene cut), segmen zoom-out tetap otomatis pindah ke
        Fit-Blur dan kembali ke Face Track begitu wajah close-up muncul lagi.
    """
    total_frames = int(total_frames)
    if total_frames <= 0:
        return []
    bounds = sorted({int(c) for c in cut_frames if 0 < int(c) < total_frames})
    starts = [0] + bounds
    ends = bounds + [total_frames]

    raw_segments = []

    def _push(s_f, e_f, mode):
        s_f, e_f = int(s_f), int(e_f)
        if e_f <= s_f:
            return
        if raw_segments and raw_segments[-1]["mode"] == mode and raw_segments[-1]["end"] == s_f:
            raw_segments[-1]["end"] = e_f
        else:
            raw_segments.append({"start": s_f, "end": e_f, "mode": mode})

    enter_fit_streak = 4   # ~0.67s wide/tanpa wajah -> masuk fit_blur
    exit_fit_streak = 3    # ~0.50s wajah close-up -> balik ke smart_crop

    for s, e in zip(starts, ends, strict=True):
        shot_pairs = [
            (idx, _is_sample_wide_or_no_face(m, ana_w))
            for idx, m in zip(sample_indices, sample_meta, strict=True)
            if s <= idx < e
        ]
        if not shot_pairs:
            _push(s, e, "fit_blur")
            continue

        # Shot pendek (< 4 sampel): pakai mayoritas langsung
        if len(shot_pairs) < enter_fit_streak:
            wide_count = sum(1 for _, is_w in shot_pairs if is_w)
            _push(s, e, "fit_blur" if wide_count * 2 >= len(shot_pairs) else "smart_crop")
            continue

        # Shot panjang: inisialisasi mode awal dari mayoritas 3 sampel pertama agar
        # 1 frame blur tepat di cut tidak memicu flash fit_blur 0.5 detik.
        head_k = min(3, len(shot_pairs))
        head_wide = sum(1 for _, is_w in shot_pairs[:head_k] if is_w)
        cur_mode = "fit_blur" if head_wide * 2 > head_k else "smart_crop"
        seg_start = s
        streak = 0
        pending_start = s

        for idx, is_wide in shot_pairs:
            want_mode = "fit_blur" if is_wide else "smart_crop"
            if want_mode == cur_mode:
                streak = 0
                continue
            if streak == 0:
                pending_start = idx
            streak += 1
            needed = enter_fit_streak if want_mode == "fit_blur" else exit_fit_streak
            if streak >= needed:
                if pending_start - seg_start >= 10:
                    _push(seg_start, pending_start, cur_mode)
                    seg_start = pending_start
                cur_mode = want_mode
                streak = 0

        _push(seg_start, e, cur_mode)

    return raw_segments

def _trajectory_cache_key(video_path, start_time, end_time):
    """Cache key from the exact (version, realpath, start, end) used by the renderer."""
    real = os.path.realpath(video_path)
    h = hashlib.sha1(
        f"{TRAJECTORY_ALGO_VERSION}|{real}|{round(float(start_time), 3)}|{round(float(end_time), 3)}".encode("utf-8")
    ).hexdigest()[:16]
    return h, os.path.join(TEMP_DIR, f"trajectory_{h}.json")

def get_or_compute_smart_crop_trajectory(video_path, start_time, end_time, job_id=None, tracking_engine=None):
    """
    Kembalikan trajektori smart crop, memakai cache JSON berbasis file supaya
    preview (server utama) dan render (worker subprocess) memakai nilai yang
    PERSIS sama. Inferensi YOLOv8/YuNet bersifat non-deterministik, maka
    trajektori wajib dihitung satu kali lalu dipakai bersama.
    """
    _, cache_path = _trajectory_cache_key(video_path, start_time, end_time)
    with _TRAJECTORY_CACHE_LOCK:
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("positions") is not None:
                    return cached
            except Exception:
                pass

        trajectory = compute_smart_crop_trajectory(
            video_path, start_time, end_time, job_id=job_id, tracking_engine=tracking_engine
        )
        # Konversi ke int EVEN, sama persis dengan _write_crop_commands, supaya
        # posisi yang dipreview == posisi crop di render final.
        positions = [(int(round(float(x))) // 2) * 2 for x in trajectory["positions"]]
        payload = {
            "fps": float(trajectory["fps"]),
            "first_pts": float(trajectory.get("first_pts") if trajectory.get("first_pts") is not None else start_time),
            "orig_w": int(trajectory["orig_w"]),
            "orig_h": int(trajectory["orig_h"]),
            "crop_w": int(trajectory["crop_w"]),
            "crop_h": int(trajectory["crop_h"]),
            "crop_y": int(trajectory["crop_y"]),
            "positions": positions,
            "cut_frames": [int(c) for c in trajectory.get("cut_frames", [])],
            "shot_modes": trajectory.get("shot_modes") or [],
            "fit_blur_zoom": float(
                trajectory["fit_blur_zoom"]
                if trajectory.get("fit_blur_zoom") is not None
                else FIT_BLUR_ZOOM_DEFAULT
            ),
            "engine": str(trajectory.get("engine") or "auto"),
        }
        tmp_path = cache_path + f".{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, cache_path)
        return payload

def compute_smart_crop_trajectory(video_path, start_time, end_time, job_id=None, tracking_engine=None):
    """
    Pass 1 Turbo Cinema Clipper: deteksi engine yang tersedia secara otomatis (YuNet ONNX / YOLOv8 / Haar),
    scan narasumber di frame resolusi kecil (640p), lalu hasilkan trajektori posisi crop 9:16 per frame
    (koordinat resolusi ASLI) yang menyesuaikan rasio/komposisi video asli, bebas distraksi objek lain,
    dan tidak pernah jatuh buta ke tengah meja kosong.
    """
    orig_w, orig_h, fps = _probe_video_stream(video_path)
    duration = max(0.1, float(end_time) - float(start_time))
    total_clip_frames = max(1, int(round(duration * fps)))
    first_pts = math.ceil(float(start_time) * fps - 1e-4) / fps

    # Ukuran crop vertikal 9:16 di resolusi asli
    target_aspect_ratio = 9.0 / 16.0
    crop_h = orig_h
    crop_w = int(crop_h * target_aspect_ratio)
    if crop_w % 2 != 0: crop_w -= 1
    if crop_w > orig_w:
        crop_w = orig_w
        if crop_w % 2 != 0: crop_w -= 1
        crop_h = int(crop_w / target_aspect_ratio)
        if crop_h % 2 != 0: crop_h -= 1
    crop_y = max(0, ((orig_h - crop_h) // 2) // 2 * 2)

    orig_aspect = orig_w / float(max(1, orig_h))
    is_native_vertical = orig_aspect <= 0.62 and (orig_w - crop_w) <= 4 and (orig_h - crop_h) <= 4
    is_portrait_or_square = orig_aspect <= 1.08

    # Jika video sumber sudah vertikal ~9:16 asli, pertahankan 100% frame asli (tanpa crop/fit-blur)
    if is_native_vertical:
        log(
            f"Video sumber sudah vertikal ({orig_w}x{orig_h}) -> memakai Native Passthrough tanpa re-crop.",
            job_id=job_id,
        )
        return {
            "fps": fps,
            "first_pts": first_pts,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "crop_y": crop_y,
            "positions": np.zeros(total_clip_frames, dtype=np.float64),
            "cut_frames": [],
            "shot_modes": [{"start": 0, "end": total_clip_frames, "mode": "smart_crop"}],
            "fit_blur_zoom": 1.0,
            "engine": "native_vertical",
            "engine_hits": {"yunet": 0, "yolo": 0, "haar": 0},
        }

    # Resolusi analisis (downscaled)
    max_ana_w = int(env_float("VC_ANALYSIS_MAX_WIDTH", ANALYSIS_MAX_WIDTH, 320, 1280))
    ana_w = min(orig_w, max_ana_w)
    ana_w -= ana_w % 2
    ana_h = max(2, int(round(orig_h * ana_w / orig_w)))
    ana_h -= ana_h % 2
    scale_x = orig_w / float(ana_w)
    scale_y = orig_h / float(ana_h)

    # Parameter berbasis WAKTU agar perilaku sama di 24/30/60 fps
    stride = max(1, int(round(fps / 6.0)))          # Jalankan AI ~6x per detik
    base_sigma = max(1.0, fps * (14.0 / 30.0))      # ~0.47 detik kurva gimbal responsif (anti-lag)
    deadzone_ratio = 0.045                          # 4.5% lebar layar = stabil tripod saat duduk, responsif saat bergeser
    deadzone_px = orig_w * deadzone_ratio
    scene_threshold = env_float("VC_SCENE_CUT_THRESHOLD", SCENE_CUT_DEFAULT_THRESHOLD, 5.0, 100.0)
    cut_detector = SceneCutDetector(fps, threshold=scene_threshold)

    engine_plan = select_tracking_engine(preferred=tracking_engine, job_id=job_id)
    primary = engine_plan["primary"]
    fallback = engine_plan["fallback"]
    device = engine_plan["device"]

    # Inisialisasi detector sesuai rencana (YOLOv8 hanya di-load di awal jika menjadi primary;
    # jika berperan sebagai fallback di CPU, YOLOv8 di-load secara LAZY hanya saat dibutuhkan)
    yunet_detector = _get_yunet_detector(ana_w, ana_h, score_threshold=0.55) if "yunet" in (primary, fallback) else None
    yolo_model = _get_yolo_model(job_id=job_id) if primary == "yolo" else None
    haar_detectors = _get_haar_detectors() if "haar" in (primary, fallback) else None

    if primary == "yolo" and yolo_model is None and fallback:
        primary, fallback = fallback, None
        if primary == "yunet" and yunet_detector is None:
            yunet_detector = _get_yunet_detector(ana_w, ana_h, score_threshold=0.55)
    elif primary == "yunet" and yunet_detector is None and fallback:
        primary, fallback = fallback, None
        if primary == "yolo" and yolo_model is None:
            yolo_model = _get_yolo_model(job_id=job_id)

    fallback_label = f" + fallback {fallback.upper()}" if fallback else ""
    log(
        f"Applying Turbo Cinema Clipper (Engine: {primary.upper()}{fallback_label} [{device.upper()}] | "
        f"Stride {stride} + Scene-Cut Aware Adaptive 9:16) | "
        f"Sumber {orig_w}x{orig_h} @ {fps:.2f}fps -> crop {crop_w}x{crop_h}",
        job_id=job_id,
    )
    if crop_w < 1080:
        log(f"Info: lebar crop {crop_w}px < 1080px, video akan di-upscale. Sumber resolusi lebih tinggi = hasil lebih tajam.", job_id=job_id)

    yolo_lazy_min_misses = 1 if device in ("cuda", "mps") else 2
    yolo_lazy_logged = False
    miss_streak = 0
    engine_hits = {"yunet": 0, "yolo": 0, "haar": 0}
    mouth_tracker = _MouthActivityTracker(ana_w)
    target_selector = _SpeakerTargetSelector(ana_w)
    speaker_switch_events = []

    def _run_detector(eng, frame_bgr):
        nonlocal yolo_model, yolo_lazy_logged
        if eng == "yunet" and yunet_detector is not None:
            dets = _detect_faces_yunet(
                yunet_detector, frame_bgr, ana_w, ana_h,
                min_w_ratio=0.04, max_cy_ratio=0.86, min_cy_ratio=0.10,
            )
            # Area meja bawah (cy > 75%) hanya diizinkan jika berupa facecam pojok (is_corner_pip)
            return [d for d in dets if float(d.get("cy", 0.0)) <= ana_h * 0.75 or d.get("is_corner_pip", False)]
        if eng == "yolo":
            if yolo_model is None:
                if not yolo_lazy_logged:
                    log("Info: mengaktifkan fallback YOLOv8 person tracking untuk sudut wajah yang terhalang...", job_id=job_id)
                    yolo_lazy_logged = True
                yolo_model = _get_yolo_model(job_id=job_id)
            if yolo_model is not None:
                return _detect_persons_yolo(
                    yolo_model, frame_bgr, ana_w, ana_h, device=device,
                    min_w_ratio=0.10, yunet_detector=yunet_detector,
                )
        if eng == "haar" and haar_detectors:
            return _detect_faces_haar(haar_detectors, frame_bgr, ana_w, ana_h, min_w_ratio=0.04)
        return []

    # ═══════════════════════════════════════════════════════════════
    # PASS 1: TURBO SCANNING & TRAJECTORY INTERPOLATION
    # ═══════════════════════════════════════════════════════════════
    sample_indices = []
    sample_positions = []  # None = deteksi gagal (ditahan per shot oleh _build_shot_trajectory)
    sample_cys = []
    sample_meta = []
    frame_idx = 0
    start_pass1_time = time.time()
    last_log_time = 0

    log(f"[Pass 1/2] Fast Scanning narasumber (~{total_clip_frames} frames @ {ana_w}x{ana_h}, stride={stride})...", job_id=job_id)

    for frame in _iter_ffmpeg_frames(video_path, start_time, duration, fps, ana_w, ana_h):
        is_cut = cut_detector.update(frame_idx, frame)
        if is_cut:
            miss_streak = 0
            mouth_tracker.reset()
            target_selector.reset()
        # Di frame cut, deteksi DIPAKSA jalan agar posisi wajah di angle baru langsung diketahui
        if is_cut or frame_idx % stride == 0:
            current_target_x = None
            detections = _run_detector(primary, frame)

            if not detections and fallback:
                miss_streak += 1
                should_try_fallback = (
                    fallback != "yolo"
                    or yolo_model is not None
                    or is_cut
                    or frame_idx == 0
                    or miss_streak >= yolo_lazy_min_misses
                )
                if should_try_fallback:
                    detections = _run_detector(fallback, frame)

            if detections:
                miss_streak = 0
                mouth_tracker.update(detections)
                best, switched_far = target_selector.select(detections)
                if best is not None:
                    if not best.get("is_corner_pip", False):
                        current_target_x = best["cx"] * scale_x
                        if best.get("cy") is not None:
                            sample_cys.append(float(best["cy"]))
                    hit_eng = best.get("engine")
                    if hit_eng in engine_hits:
                        engine_hits[hit_eng] += 1
                    if switched_far and not is_cut and current_target_x is not None:
                        speaker_switch_events.append({
                            "sample_k": len(sample_indices),
                            "frame_idx": frame_idx,
                            "delay_samples": max(1, int(target_selector.last_switch_delay or 1)),
                        })

            sample_indices.append(frame_idx)
            sample_positions.append(current_target_x)
            if detections:
                cxs = [float(d["cx"]) for d in detections]
                ws = [float(d.get("w", 0.0)) for d in detections]
                sample_meta.append({
                    "count": len(detections),
                    "min_cx": min(cxs),
                    "max_cx": max(cxs),
                    "min_w": min(ws),
                    "max_w": max(ws),
                    "is_face": any(d.get("engine") in ("yunet", "haar") for d in detections),
                    "is_corner_pip": all(bool(d.get("is_corner_pip", False)) for d in detections),
                })
            else:
                sample_meta.append({
                    "count": 0, "min_cx": None, "max_cx": None, "min_w": 0.0, "max_w": 0.0,
                    "is_face": False, "is_corner_pip": False,
                })

        frame_idx += 1

        curr_time = time.time()
        if curr_time - last_log_time >= 1.5:
            percent = min(100.0, (frame_idx / total_clip_frames) * 100)
            elapsed = curr_time - start_pass1_time
            eta = (elapsed / (percent / 100) - elapsed) if percent > 0 else 0
            eta_str = format_timestamp(eta).split('.')[0]
            log(f"Progress Pass 1 (Turbo Scan): {percent:.1f}% ({frame_idx}/{total_clip_frames}) | ETA: {eta_str}", job_id=job_id)
            last_log_time = curr_time

    if frame_idx == 0:
        raise RuntimeError(f"Gagal decode frame video untuk tracking: {video_path}")

    if not sample_indices:
        sample_indices = [0]
        sample_positions = [None]
        sample_meta = [{
            "count": 0, "min_cx": None, "max_cx": None, "min_w": 0.0, "max_w": 0.0,
            "is_face": False, "is_corner_pip": False,
        }]

    # Jika tinggi crop < tinggi video asli (mis. video 4:5 / 3:4), sesuaikan crop_y ke tinggi wajah asli
    if crop_h < orig_h and sample_cys:
        med_cy_orig = float(np.median(sample_cys)) * scale_y
        adaptive_y = int(round(med_cy_orig - crop_h * 0.38))
        crop_y = max(0, min(orig_h - crop_h, (adaptive_y // 2) * 2))

    cut_frames = list(cut_detector.cuts)
    if cut_frames:
        cut_times = ", ".join(f"{c / fps:.1f}s" for c in cut_frames[:12])
        more = " ..." if len(cut_frames) > 12 else ""
        log(f"Terdeteksi {len(cut_frames)} pergantian kamera -> {len(cut_frames) + 1} shot (di {cut_times}{more})", job_id=job_id)
    else:
        log("Tidak ada pergantian kamera terdeteksi (1 shot).", job_id=job_id)

    if is_portrait_or_square:
        shot_modes = [{"start": 0, "end": frame_idx, "mode": "smart_crop"}]
    else:
        shot_modes = _classify_shot_modes(frame_idx, cut_frames, sample_indices, sample_meta, ana_w)

    # Backfill aman + Virtual Cut untuk pergantian pembicara di dalam satu shot:
    # 1) Jika switch terjadi <= 1.25s setelah cut (warm-up EMA bibir), backfill hanya ke awal shot itu.
    # 2) Jika switch terjadi di tengah percakapan panjang, hanya backfill sampel delay konfirmasi (~0.6s)
    #    dan tambahkan Virtual Cut agar kamera langsung pindah tanpa menyapu ruang tengah meja!
    warmup_frames = max(stride * 7, int(round(fps * 1.25)))
    cut_set = set(cut_frames)
    effective_cuts = list(cut_frames)
    for ev in speaker_switch_events:
        k = ev["sample_k"]
        s_idx = ev["frame_idx"]
        delay = ev["delay_samples"]
        if k >= len(sample_positions) or s_idx in cut_set or sample_positions[k] is None:
            continue
        shot_start = max((c for c in cut_frames if c <= s_idx), default=0)
        if (s_idx - shot_start) <= warmup_frames:
            for j in range(k - 1, -1, -1):
                if sample_indices[j] < shot_start:
                    break
                sample_positions[j] = sample_positions[k]
        else:
            onset_k = k
            for j in range(k - 1, max(-1, k - delay) - 1, -1):
                if j < 0 or sample_indices[j] <= shot_start:
                    break
                sample_positions[j] = sample_positions[k]
                onset_k = j
            onset_frame = sample_indices[onset_k]
            if 0 < onset_frame < frame_idx:
                effective_cuts.append(onset_frame)

    effective_cuts = sorted(set(effective_cuts))

    # Hapus lonjakan posisi singkat (1-3 sampel / ~0.16-0.5s) akibat tangan/gelas/objek lewat
    _suppress_short_burst_spikes(sample_indices, sample_positions, effective_cuts, orig_w, max_burst=3)

    # Hitung zoom Fit-Blur adaptif (1.0x untuk B-roll/slide/PiP; 1.0-1.52x dinamis agar 2 pembicara utuh)
    explicit_zoom = os.environ.get("VC_FIT_BLUR_ZOOM")
    adaptive_zoom = _compute_adaptive_fit_blur_plan(
        shot_modes, sample_indices, sample_meta, sample_positions, ana_w, orig_w
    )
    if explicit_zoom is not None and str(explicit_zoom).strip() != "":
        fit_blur_zoom = env_float("VC_FIT_BLUR_ZOOM", adaptive_zoom, 1.0, 2.5)
    else:
        fit_blur_zoom = adaptive_zoom

    fit_blur_shots = sum(1 for m in shot_modes if m["mode"] == "fit_blur")
    if fit_blur_shots:
        log(
            f"Hybrid Framing: {fit_blur_shots} segmen wide/B-roll menyesuaikan video asli dengan Fit-Blur (zoom {fit_blur_zoom:.2f}x).",
            job_id=job_id,
        )

    # Guard split-screen (HANYA pada segmen smart_crop 9:16): jangan biarkan crop sempit jatuh ke garis tengah
    def _is_in_fit_blur(f_idx):
        return any(m["mode"] == "fit_blur" and m["start"] <= f_idx < m["end"] for m in shot_modes)

    seam_min_x = orig_w * 0.42
    seam_max_x = orig_w * 0.58
    for k, meta in enumerate(sample_meta):
        if _is_in_fit_blur(sample_indices[k]):
            continue
        if (
            meta["count"] >= 2
            and meta["min_cx"] is not None
            and meta["max_cx"] is not None
            and (meta["max_cx"] - meta["min_cx"]) > ana_w * 0.30
        ):
            pos_x = sample_positions[k]
            if pos_x is None or (seam_min_x <= pos_x <= seam_max_x):
                left_x = meta["min_cx"] * scale_x
                right_x = meta["max_cx"] * scale_x
                if pos_x is None:
                    sample_positions[k] = left_x
                else:
                    sample_positions[k] = left_x if abs(pos_x - left_x) <= abs(pos_x - right_x) else right_x

    # Global Speaker Anchor: hindari fallback ke orig_w / 2.0 (tengah meja kosong) jika ada pembicara di klip
    valid_positions = [float(p) for p in sample_positions if p is not None]
    anchor_default_x = valid_positions[0] if valid_positions else (orig_w / 2.0)

    # Jika di dalam satu shot pembicara bergerak aktif (mis. vlog jalan kaki / handheld pan > 18% layar),
    # gunakan kurva smoothing lebih cepat agar kamera tidak tertinggal (lagging) dari video asli.
    smooth_sigma = base_sigma
    if len(valid_positions) >= 4:
        shot_spans = []
        bounds = [0] + [c for c in effective_cuts if 0 < c < frame_idx] + [frame_idx]
        for s_b, e_b in zip(bounds[:-1], bounds[1:], strict=True):
            sv = [
                float(p) for idx_s, p in zip(sample_indices, sample_positions, strict=True)
                if s_b <= idx_s < e_b and p is not None and not _is_in_fit_blur(idx_s)
            ]
            if len(sv) >= 3:
                shot_spans.append(max(sv) - min(sv))
        if shot_spans and max(shot_spans) > orig_w * 0.18:
            smooth_sigma = max(1.0, fps * (9.0 / 30.0))

    smooth_center_x = _build_shot_trajectory(
        frame_idx, sample_indices, sample_positions, effective_cuts,
        deadzone_px=deadzone_px, sigma=smooth_sigma, default_x=anchor_default_x,
    )

    # Konversi target center ke sudut kiri atas (X1) & clamp batas video
    smooth_crop_x = smooth_center_x - (crop_w / 2.0)
    smooth_crop_x = np.clip(smooth_crop_x, 0, orig_w - crop_w)

    hits_summary = ", ".join(f"{k}={v}" for k, v in engine_hits.items() if v > 0) or "none"
    log(f"[Pass 1/2] Tracking selesai ({frame_idx} frames, deteksi: {hits_summary}).", job_id=job_id)
    return {
        "fps": fps,
        "first_pts": first_pts,
        "orig_w": orig_w,
        "orig_h": orig_h,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "crop_y": crop_y,
        "positions": smooth_crop_x,
        "cut_frames": effective_cuts,
        "shot_modes": shot_modes,
        "fit_blur_zoom": fit_blur_zoom,
        "engine": primary,
        "engine_hits": engine_hits,
    }


def _escape_filter_path(path):
    """Escape file path for use inside an FFmpeg filtergraph option value."""
    return path.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")


def _write_crop_commands(trajectory, cmd_path):
    """
    Tulis trajektori crop sebagai file sendcmd FFmpeg (hanya saat posisi berubah).
    Return posisi X awal untuk inisialisasi filter crop.
    """
    fps = float(trajectory["fps"])
    positions = trajectory["positions"]
    first_x = None
    last_x = None
    with open(cmd_path, "w", encoding="utf-8") as f:
        for i, x in enumerate(positions):
            xi = (int(round(float(x))) // 2) * 2
            if first_x is None:
                first_x = xi
            if xi == last_x:
                continue
            # Setengah frame lebih awal: pembulatan PTS tidak membuat crop telat 1 frame di titik cut
            t = 0.0 if i == 0 else max(0.0, (i - 0.5) / fps)
            f.write(f"{t:.4f} crop x {xi};\n")
            last_x = xi
    return first_x or 0


def build_smart_crop_chain(trajectory, cmd_path):
    """
    Bangun filter chain FFmpeg yang melakukan crop dinamis (via sendcmd) + scale ke 1080x1920.
    Semua dilakukan di dalam satu proses FFmpeg -> hanya 1x encode di render final.
    """
    x0 = _write_crop_commands(trajectory, cmd_path)
    fps = float(trajectory.get("fps") or 30.0)
    return (
        f"setpts=PTS-STARTPTS,fps={fps:.6f},"
        f"sendcmd=f='{_escape_filter_path(cmd_path)}',"
        f"crop={trajectory['crop_w']}:{trajectory['crop_h']}:{x0}:{trajectory['crop_y']},"
        f"scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos+accurate_rnd+full_chroma_int,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    )


def _write_fit_blur_commands(trajectory, fg_scaled_w, fit_cmd_path, fit_segments=None):
    """Tulis trajektori crop X per frame untuk foreground Fit-Blur 1.78x via sendcmd FFmpeg."""
    fps = float(trajectory.get("fps") or 30.0)
    orig_w = float(trajectory.get("orig_w") or 1920)
    crop_w = float(trajectory.get("crop_w") or 606)
    positions = trajectory.get("positions")
    if positions is None or len(positions) == 0:
        default_x = int(round(((orig_w - crop_w) * 0.28 + crop_w / 2.0) * (fg_scaled_w / orig_w) - 540.0))
        default_x = max(0, min(max(0, fg_scaled_w - 1080), (default_x // 2) * 2))
        with open(fit_cmd_path, "w", encoding="utf-8") as f:
            f.write(f"0.0000 crop x {default_x};\n")
        return default_x

    max_fit_x = max(0, fg_scaled_w - 1080)
    scale_fit = fg_scaled_w / orig_w

    def _to_fit_x(pos):
        cx_orig = float(pos) + crop_w / 2.0
        xi = int(round(cx_orig * scale_fit - 540.0))
        return max(0, min(max_fit_x, (xi // 2) * 2))

    if fit_segments:
        fit_mask = np.zeros(len(positions), dtype=bool)
        for seg in fit_segments:
            s_i = max(0, int(seg.get("start", 0)))
            e_i = min(len(positions), int(seg.get("end", len(positions))))
            if e_i > s_i:
                fit_mask[s_i:e_i] = True
    else:
        fit_mask = np.ones(len(positions), dtype=bool)

    first_x = None
    last_x = None
    lines = []
    for i, pos in enumerate(positions):
        if not fit_mask[i]:
            continue
        xi = _to_fit_x(pos)
        if first_x is None:
            first_x = xi
            last_x = xi
            lines.append(f"0.0000 crop x {xi};\n")
            continue
        if xi == last_x:
            continue
        t = max(0.0, (i - 0.5) / fps)
        lines.append(f"{t:.4f} crop x {xi};\n")
        last_x = xi

    if first_x is None:
        first_x = _to_fit_x(positions[0])
        lines.append(f"0.0000 crop x {first_x};\n")

    with open(fit_cmd_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return first_x


def build_hybrid_smart_crop_graph(trajectory, cmd_path, finish_chain, out_label="v"):
    """
    Filtergraph hybrid Smart Crop + Fit-Blur (zoom 1.78x):
      - Shot close-up dengan wajah jelas -> smart_crop 9:16 mengikuti pembicara.
      - Shot wide/zoom-out/tanpa wajah -> Fit-Blur dengan foreground zoom 1.78x
        yang dikunci horizontal ke posisi pembicara agar tidak memotong tengah ruangan/meja.
    """
    shot_modes = trajectory.get("shot_modes") or []
    fit_segments = [m for m in shot_modes if m.get("mode") == "fit_blur"]
    if not fit_segments:
        return f"[0:v]{build_smart_crop_chain(trajectory, cmd_path)},{finish_chain}[{out_label}]"

    zoom = float(trajectory.get("fit_blur_zoom") or FIT_BLUR_ZOOM_DEFAULT)
    fps = float(trajectory.get("fps") or 30.0)
    fps_prefix = f"fps={fps:.6f},"
    total_frames = len(trajectory.get("positions") or [])
    fg_scaled_w = max(1080, (int(round(1080.0 * zoom)) // 2) * 2)
    fit_cmd_path = f"{os.path.splitext(cmd_path)[0]}_fit.txt"
    fit_x0 = _write_fit_blur_commands(trajectory, fg_scaled_w, fit_cmd_path, fit_segments=fit_segments)
    fit_fg_chain = (
        f"scale={fg_scaled_w}:-2:flags=lanczos+accurate_rnd+full_chroma_int,"
        f"sendcmd=f='{_escape_filter_path(fit_cmd_path)}',"
        f"crop=1080:min(ih\\,1920):{fit_x0}:0"
    )

    if len(fit_segments) == 1 and fit_segments[0].get("start", 0) <= 0 and fit_segments[0].get("end", 0) >= total_frames:
        return (
            f"[0:v]setpts=PTS-STARTPTS,{fps_prefix}split=2[bg_in][fg_in];"
            f"[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5,eq=brightness=-0.08[bg];"
            f"[fg_in]{fit_fg_chain}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2-(H*0.05),{finish_chain}[{out_label}]"
        )

    enable_terms = []
    for seg in fit_segments:
        s_idx = int(seg.get("start", 0))
        e_idx = int(seg.get("end", 0))
        t0 = 0.0 if s_idx <= 0 else max(0.0, (s_idx - 0.5) / fps)
        t1 = (e_idx + 1.0) / fps if e_idx >= total_frames else max(t0 + 0.01, (e_idx - 0.5) / fps)
        enable_terms.append(f"between(t,{t0:.4f},{t1:.4f})")
    enable_expr = "+".join(enable_terms)

    crop_chain = build_smart_crop_chain(trajectory, cmd_path)
    if crop_chain.startswith("setpts=PTS-STARTPTS,"):
        crop_chain = crop_chain[len("setpts=PTS-STARTPTS,"):]
    if crop_chain.startswith(fps_prefix):
        crop_chain = crop_chain[len(fps_prefix):]

    return (
        f"[0:v]setpts=PTS-STARTPTS,{fps_prefix}split=3[crop_in][bg_in][fg_in];"
        f"[crop_in]{crop_chain}[v_crop];"
        f"[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5,eq=brightness=-0.08[bg];"
        f"[fg_in]{fit_fg_chain}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2-(H*0.05)[v_fit];"
        f"[v_crop][v_fit]overlay=0:0:enable='{enable_expr}',{finish_chain}[{out_label}]"
    )


def build_fit_blur_graph(finish_chain, out_label="v"):
    """
    Filtergraph Fit 16:9 + Blurred Background.
    Main 16:9 video dinaikkan 5% ke atas (posisi seimbang & proporsional untuk subtitle).
    """
    return (
        f"[0:v]setpts=PTS-STARTPTS,split=2[bg_in][fg_in];"
        f"[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5,eq=brightness=-0.08[bg];"
        f"[fg_in]scale=1080:-2:flags=lanczos+accurate_rnd+full_chroma_int[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2-(H*0.05),{finish_chain}[{out_label}]"
    )


def build_split_screen_graph(video_path, start_time, end_time, finish_chain, job_id=None, out_label="v"):
    """
    Filtergraph Dual Stack Split-Screen (Top & Bottom 50:50).
    Top window: Left speaker / Host (1080x960)
    Bottom window: Right speaker / Guest (1080x960)
    Zero panning motion sickness, single-pass encode bersama finish_chain.
    """
    try:
        width, height, _ = _probe_video_stream(video_path)
    except Exception:
        width, height = 1920, 1080

    target_crop_w = int(height * 0.54)
    if target_crop_w % 2 != 0:
        target_crop_w -= 1

    anchors = detect_speaker_anchors(video_path, start_time, end_time, width, height, target_crop_w)
    left_cx = anchors.get("left_anchor", width * 0.25)
    right_cx = anchors.get("right_anchor", width * 0.75)

    top_x = max(0, min(int(left_cx - target_crop_w / 2), width - target_crop_w))
    bottom_x = max(0, min(int(right_cx - target_crop_w / 2), width - target_crop_w))

    top_x = (top_x // 2) * 2
    bottom_x = (bottom_x // 2) * 2

    return (
        f"[0:v]setpts=PTS-STARTPTS,split=2[vtop_in][vbot_in];"
        f"[vtop_in]crop={target_crop_w}:{height}:{top_x}:0,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[top];"
        f"[vbot_in]crop={target_crop_w}:{height}:{bottom_x}:0,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[bot];"
        f"[top][bot]vstack=inputs=2[stacked];"
        f"[stacked]drawbox=x=0:y=958:w=1080:h=4:color=0x18181b@0.9:t=fill,{finish_chain}[{out_label}]"
    )


def apply_smart_crop(video_path, start_time, end_time, output_path, job_id=None):
    """
    Turbo Cinema Clipper standalone (tanpa subtitle/grading): tracking + crop 9:16 + audio
    dalam SATU kali encode FFmpeg (tidak ada file intermediate mp4v lagi).
    """
    clip_dur = end_time - start_time
    trajectory = get_or_compute_smart_crop_trajectory(video_path, start_time, end_time, job_id=job_id)
    cmd_path = os.path.join(TEMP_DIR, f"temp_cropcmd_{int(time.time()*1000)}.txt")
    try:
        chain = build_smart_crop_chain(trajectory, cmd_path)
        log("[Pass 2/2] Rendering Video 9:16 (1 pass encode)...", job_id=job_id)
        ffmpeg_cmd = [
            get_ffmpeg_binary(), "-y",
            "-ss", str(start_time),
            "-t", str(clip_dur),
            "-i", video_path,
            "-filter_complex", f"[0:v]{chain}[v]",
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            output_path
        ]
        run_command(ffmpeg_cmd, "Rendering cinema video & audio...", job_id=job_id, duration=clip_dur)
    finally:
        if os.path.exists(cmd_path):
            try:
                os.remove(cmd_path)
            except Exception:
                pass

    log("Cinema Clipper selesai 100% dengan mulus.", job_id=job_id)
    return output_path

def _validate_rendered_video(file_path: str, min_duration: float = 0.5) -> bool:
    """Validate that rendered file exists, non-empty, contains video stream, and has valid duration."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) <= 1024:
        return False

    ffprobe_bin = get_ffprobe_binary()
    try:
        # Check stream video
        cmd_streams = [
            ffprobe_bin, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height",
            "-of", "json",
            file_path
        ]
        res_streams = subprocess.run(cmd_streams, capture_output=True, text=True)
        if res_streams.returncode != 0:
            return False
        stream_data = json.loads(res_streams.stdout)
        streams = stream_data.get("streams", [])
        if not streams or not streams[0].get("codec_name"):
            return False

        # Check format duration
        cmd_duration = [
            ffprobe_bin, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            file_path
        ]
        res_dur = subprocess.run(cmd_duration, capture_output=True, text=True)
        if res_dur.returncode != 0:
            return False
        dur_str = res_dur.stdout.strip()
        if not dur_str:
            return False
        dur = float(dur_str)
        if dur < min_duration:
            return False
        return True
    except Exception:
        return False

def _natural_padding(segments, clip_start, clip_end, lead_max=0.20, tail_max=0.50):
    """Hitung padding awal dan akhir yang adaptif dengan jeda ucapan asli.

    Mencegah memotong kata di awal dan mencegah kalimat baru bocor di akhir.
    """
    words = []
    for s in segments or []:
        w_list = s.get("words") or []
        if w_list:
            for w in w_list:
                try:
                    words.append((float(w.get("start", 0.0)), float(w.get("end", 0.0))))
                except (TypeError, ValueError):
                    pass
        else:
            try:
                words.append((float(s.get("start", 0.0)), float(s.get("end", 0.0))))
            except (TypeError, ValueError):
                pass

    if not words:
        return 0.05, 0.25

    words.sort(key=lambda x: x[0])

    # Cari kata terakhir sebelum clip_start
    prev_ends = [e for s_, e in words if e <= clip_start + 0.05]
    prev_end = max(prev_ends) if prev_ends else (clip_start - lead_max)
    gap_before = max(0.0, clip_start - prev_end)
    lead = min(lead_max, max(0.0, gap_before * 0.5))

    # Cari kata pertama setelah clip_end
    next_starts = [s_ for s_, e in words if s_ >= clip_end - 0.05]
    if next_starts:
        next_start = min(next_starts)
        gap_after = max(0.0, next_start - clip_end)
        # Jaga jarak aman minimal 40ms sebelum kata berikutnya agar suku kata tidak bocor
        safe_gap = max(0.0, gap_after - 0.04)
        tail = min(tail_max, min(safe_gap, gap_after * 0.6))
    else:
        # Tidak ada kata setelahnya di transkrip: beri tail nafas alami
        tail = min(tail_max, 0.30)

    return lead, tail


def extract_and_format_clip(video_path, clip_info, index, segments, out_dir, job_id=None, sub_style="tiktok_viral", framing_mode="smart_crop", sub_position=68, watermark_enabled=True, source_watermark="", account_watermark=""):
    raw_start = max(0.0, float(clip_info.get("start", 0.0)))
    raw_end = float(clip_info.get("end", 0.0))
    lead, tail = _natural_padding(segments, raw_start, raw_end)
    start = max(0.0, raw_start - lead)
    end = raw_end + tail
    duration = end - start
    fade_dur = max(0.12, min(0.35, tail * 0.75))
    fade_out_start = max(0.0, duration - fade_dur)
    
    title_options = clip_info.get("title_options", [clip_info.get("title", f"clip_{index}")])
    title = title_options[0] if isinstance(title_options, list) and title_options else title_options
    
    safe_title = "".join(c if c.isalnum() else "_" for c in title).strip("_")
    out_file = os.path.join(out_dir, f"{index:02d}_{safe_title}.mp4")
    part_out_file = out_file + ".part"
    caption_file = os.path.join(out_dir, f"{index:02d}_{safe_title}.txt")
    
    caption_text = clip_info.get("caption", title)
    hashtags = clip_info.get("hashtags", [])
    hashtags_text = " ".join(hashtags) if isinstance(hashtags, list) else hashtags
    titles_text = "\n".join([f"- {t}" for t in title_options]) if isinstance(title_options, list) else title_options

    # Sidecar fields the posting UI needs. `main._parse_rendered_clip_metadata`
    # reads these back, so keep the headings in sync with that parser.
    hook_line = str(clip_info.get("hook_line") or "").strip()
    first_comment = str(clip_info.get("first_comment") or "").strip()
    series = str(clip_info.get("series") or "").strip()

    with open(caption_file, "w", encoding="utf-8") as f:
        f.write(f"Title Options:\n{titles_text}\n\n")
        if hook_line:
            f.write(f"Hook:\n{hook_line}\n\n")
        f.write(f"Caption:\n{caption_text}\n\n")
        f.write(f"Hashtags:\n{hashtags_text}\n")
        if first_comment:
            f.write(f"\n\nFirst Comment:\n{first_comment}\n")
        if series:
            f.write(f"\nSeries:\n{series}\n")
    
    ass_file = os.path.join(TEMP_DIR, f"temp_{index}.ass")
    if os.path.exists(ass_file):
        try:
            os.remove(ass_file)
        except Exception:
            pass
    generate_ass_subtitles(
        segments,
        ass_file,
        start,
        end,
        sub_style=sub_style,
        sub_position=sub_position,
        watermark_enabled=watermark_enabled,
        source_watermark=source_watermark,
        account_watermark=account_watermark,
    )
    crop_cmd_file = os.path.join(TEMP_DIR, f"temp_cropcmd_{index}_{int(time.time()*1000)}.txt")
    
    escaped_ass = _escape_filter_path(ass_file)
    
    af_filter = f"asetpts=PTS-STARTPTS,highpass=f=100,acompressor=threshold=-20dB:ratio=4:attack=5:release=50:makeup=8dB,afade=t=out:st={fade_out_start:.3f}:d={fade_dur:.3f}"
    # Audio Enhancer: highpass + compressor halus + loudnorm 2-pass ke target LUFS platform (default -14)
    audio_settings = get_audio_settings()
    audio_pre_chain = build_audio_pre_chain(denoise=audio_settings["denoise"])
    loudness = _measure_loudness(video_path, start, duration, audio_pre_chain, audio_settings["target_lufs"], job_id=job_id)
    af_filter, audio_mode = build_audio_filter(fade_out_start, audio_pre_chain, audio_settings["target_lufs"], loudness, fade_dur=fade_dur)
    if audio_mode == "two_pass":
        log(f"Audio: loudnorm 2-pass {loudness['input_i']:.1f} LUFS -> {audio_settings['target_lufs']:g} LUFS "
            f"(true peak {LOUDNORM_TRUE_PEAK:g} dBTP){' + denoise' if audio_settings['denoise'] else ''}", job_id=job_id)
    elif audio_mode == "skip":
        log("Audio: klip (hampir) sunyi, normalisasi volume dilewati.", job_id=job_id)
    else:
        log("Audio: pengukuran loudness gagal, memakai loudnorm 1-pass.", job_id=job_id)
    
    codecs_to_try = []
    tiktok_golden = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p", "-g", "60", "-bf", "2", "-crf", "18", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart"]
    nvenc_golden = ["-c:v", "h264_nvenc", "-preset", "slow", "-profile:v", "high", "-rc", "constqp", "-qp", "18", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709", "-g", "60", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart"]
    
    if sys.platform == "darwin":
        codecs_to_try = [
            ("VideoToolbox (TikTok Golden)", ["-c:v", "h264_videotoolbox", "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "60", "-b:v", "10M", "-maxrate", "12M", "-bufsize", "20M", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart"]),
            ("libx264 (TikTok Golden)", tiktok_golden)
        ]
    else:
        codecs_to_try = [
            ("NVENC (TikTok Golden)", nvenc_golden),
            ("Safer NVENC", ["-c:v", "h264_nvenc", "-preset", "p2", "-profile:v", "main", "-b:v", "8M", "-c:a", "aac", "-b:a", AUDIO_BITRATE]),
            ("libx264 (TikTok Golden)", tiktok_golden)
        ]

    # Grading + subtitle + fade diterapkan di filtergraph yang SAMA dengan crop/layout,
    # sehingga seluruh klip hanya di-encode satu kali (tanpa file intermediate).
    # Subtitle di-render di ruang warna RGB24 (full chroma 4:4:4) lalu dikonversi ke yuv420p
    # dengan lanczos+accurate_rnd+full_chroma_int+full_chroma_inp agar tepi huruf & warna highlight
    # tidak pecah/burem akibat chroma subsampling 4:2:0 default.
    grade_chain = "eq=contrast=1.1:brightness=0.02:saturation=1.2,unsharp=5:5:0.4:5:5:0.0"
    has_subtitles = os.path.exists(ass_file) and os.path.getsize(ass_file) > 0
    escaped_fonts_dir = _escape_filter_path(FONTS_DIR) if (FONTS_DIR and os.path.exists(FONTS_DIR)) else ""
    fonts_opt = f":fontsdir='{escaped_fonts_dir}'" if escaped_fonts_dir else ""
    subtitle_filter = (
        f",format=rgb24,subtitles='{escaped_ass}'{fonts_opt},"
        f"scale=flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,format=yuv420p"
        if has_subtitles else ""
    )
    subtitle_stage = f"{grade_chain}{subtitle_filter}"
    fade_filter = f"fade=t=out:st={fade_out_start:.3f}:d={fade_dur:.3f}"

    def _assemble(prefix_graph):
        """Tempel fade di belakang graph layout/crop/subtitle."""
        return f"{prefix_graph};[vsub]{fade_filter}[v]"

    ffmpeg_bin = get_ffmpeg_binary()

    def build_cmd(filter_graph, codec_params):
        return [
            ffmpeg_bin, "-y",
            "-accurate_seek",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", video_path,
            "-filter_complex", filter_graph,
            "-map", "[v]",
            "-map", "0:a?",
            "-af", af_filter,
            "-ar", str(AUDIO_SAMPLE_RATE),  # Wajib: loudnorm internal 192kHz
            "-avoid_negative_ts", "make_zero",
            "-t", f"{duration:.3f}",
        ] + codec_params + ["-f", "mp4", part_out_file]

    def try_render(filter_graph, label):
        for name, codec_params in codecs_to_try:
            if os.path.exists(part_out_file):
                try:
                    os.remove(part_out_file)
                except Exception:
                    pass
            try:
                run_command(build_cmd(filter_graph, codec_params), f"{label} ({name})...", job_id=job_id, duration=duration)
                if _validate_rendered_video(part_out_file, min_duration=max(0.3, duration * 0.7)):
                    os.replace(part_out_file, out_file)
                    return True
                log(f"{name} produced an invalid or unreadable video file.", job_id=job_id)
            except Exception as e:
                log(f"{name} Encoding failed: {e}. Falling back to next codec...", job_id=job_id)
        return False

    render_success = False
    primary_graph = None
    # Semua graph layout berakhir di label [vsub] (hasil subtitle) supaya fade
    # bisa disambung dari titik itu.
    try:
        if framing_mode == "fit_blur":
            log(f"Applying 16:9 Fit Blur Background for Clip {index} (Podcast/Live stream mode)...", job_id=job_id)
            primary_graph = _assemble(build_fit_blur_graph(subtitle_stage, out_label="vsub"))
        elif framing_mode == "split_screen":
            log(f"Applying Studio Dual-Stack Split-Screen (50:50 Host/Guest) for Clip {index}...", job_id=job_id)
            primary_graph = _assemble(
                build_split_screen_graph(video_path, start, end, subtitle_stage,
                                         job_id=job_id, out_label="vsub")
            )
        else:
            log(f"Applying AI Smart Face Tracking 9:16 for Clip {index}...", job_id=job_id)
            trajectory = get_or_compute_smart_crop_trajectory(video_path, start, end, job_id=job_id)
            primary_graph = _assemble(
                build_hybrid_smart_crop_graph(trajectory, crop_cmd_file, subtitle_stage, out_label="vsub")
            )
    except Exception as e:
        log(f"Face tracking error, using full-frame Fit-Blur fallback to preserve original video: {e}", job_id=job_id)
        primary_graph = None

    fallback_graph = _assemble(build_fit_blur_graph(subtitle_stage, out_label="vsub"))

    crop_fit_cmd_file = f"{os.path.splitext(crop_cmd_file)[0]}_fit.txt"
    try:
        if primary_graph:
            log(f"[Pass 2/2] Single-pass render (Crop + Subtitles + Audio Enhancer + Color Grading) for Clip {index}: {title}", job_id=job_id)
            render_success = try_render(primary_graph, "Rendering final clip")
            if not render_success:
                log(f"Render utama gagal untuk Clip {index}, mencoba Podcast Dual-Anchor fallback...", job_id=job_id)
        if not render_success:
            render_success = try_render(fallback_graph, f"Fallback processing for Clip {index}")
    finally:
        for tmp in [ass_file, crop_cmd_file, crop_fit_cmd_file, part_out_file]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as e:
                log(f"Warning: Could not remove temp file {tmp}: {e}", job_id=job_id)

    if not render_success or not os.path.exists(out_file):
        raise RuntimeError(f"All codecs failed to render clip {index} ('{title}'). Output file was not generated.")

    # Emit structured output marker for frontend / social integration
    clip_output_data = {
        "index": index,
        "title": title,
        "out_file": os.path.abspath(out_file),
        "caption": caption_text,
        "hashtags": hashtags if isinstance(hashtags, list) else [hashtags]
    }
    print(f"[CLIP_RENDERED]{json.dumps(clip_output_data)}", flush=True)

    return out_file

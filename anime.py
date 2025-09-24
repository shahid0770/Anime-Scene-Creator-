import os
import re
import json
import time
import tempfile
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, vfx, transfx

# Optional: AWS Polly (graceful fallback)
try:
    import boto3
    POLLY_AVAILABLE = True
except Exception:
    POLLY_AVAILABLE = False

# Constants (configurable)
CANVAS = (1080, 1920)  # Default resolution
CHAR_STYLE_BANK = [
    "cool teenage swordsman, black hair, navy jacket, tech core, stylish boots",
    "soft-spoken strategist, brown hair, scarf, school uniform remix, subtle glow",
    "energetic prankster, spiky red hair, streetwear, fingerless gloves, neon accents",
    "mysterious transfer student, silver hair, long coat, calm intense eyes",
    "cheerful artist, teal hair, headphones, hoodie with patches, messenger bag"
]
BG_SUGGESTIONS = [
    "enchanted forest at sunset, glowing fireflies",
    "futuristic city street at night, neon rain",
    "quiet school rooftop at golden hour",
    "cozy classroom morning sunlight rays",
    "abandoned train platform with drifting fog"
]
VOICE_POOLS = {
    "en-US": ["Joanna", "Matthew", "Ivy", "Justin", "Salli", "Kendra"],
    "en-GB": ["Amy", "Brian", "Emma"],
    "hi-IN": ["Aditi", "Kajal"] if POLLY_AVAILABLE else [],
    "ja-JP": ["Mizuki", "Takumi"]
}

# UI Theme
st.set_page_config(page_title="🎬 Anime Shorts Studio", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 20% -10%, #1a1b2e 30%, #0d0f1a 70%), linear-gradient(120deg,#0f1322,#0b0d18);
  color: #e6e6f0;
}
h1, h2, h3, h4 { color: #f6f6ff; }
.block-container { padding-top: 1rem; }
.stButton>button { background: #6e6bff; border: 0; color: #fff; border-radius: 14px; padding: .55rem 1rem; }
.stDownloadButton>button { background: #1fdf64; border: 0; color: #001; border-radius: 14px; padding: .55rem 1rem; font-weight: 600;}
.badge { display:inline-block; padding: 4px 10px; border-radius: 999px; background: #262a40; border: 1px solid #3b4060; color: #cfd3ff; font-size: 12px; }
.card { background:#101329; border:1px solid #2b2f4a; border-radius:16px; padding:16px; }
input, textarea, .stTextInput>div>div>input, .stTextArea textarea { background:#0f1328 !important; color:#eaeaff !important; border-radius:12px !important; }
.progress-bar { height: 8px; border-radius: 4px; background: #262a40; margin: 12px 0; }
.progress-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #6e6bff, #9c7aff); transition: width 0.3s ease; }
.high-contrast [data-testid="stAppViewContainer"] { background: #ffffff; color: #000000; }
.high-contrast .stButton>button { background: #0000ff; color: #ffffff; }
.high-contrast .stDownloadButton>button { background: #00ff00; color: #000000; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'progress' not in st.session_state:
    st.session_state.progress = 0
    st.session_state.status = "Ready to start"
    st.session_state.current_step = 0

def update_progress(step: int, total_steps: int, status: str):
    """Update progress bar and status"""
    st.session_state.progress = int((step / total_steps) * 100)
    st.session_state.status = status
    st.session_state.current_step = step

# Image Utilities
def safe_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load font with fallback"""
    font_map = {
        "DejaVuSans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "LiberationSans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "Arial": "Arial.ttf",
        "AnimeAce": "AnimeAce.ttf",
    }
    path = font_map.get(name, font_map["DejaVuSans"])
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def remove_white_bg(image: Image.Image, thr: int = 225) -> Image.Image:
    """Make near-white pixels transparent"""
    img = image.convert("RGBA")
    data = np.array(img)
    r, g, b, a = data.T
    white_mask = (r > thr) & (g > thr) & (b > thr)
    data[..., 3][white_mask.T] = 0
    out = Image.fromarray(data)
    out = out.filter(ImageFilter.GaussianBlur(0.6))
    return out

def add_drop_shadow(char_rgba: Image.Image, offset=(14, 16), blur=18, opacity=140) -> Image.Image:
    """Add blurred shadow to character"""
    char = char_rgba.convert("RGBA")
    alpha = char.split()[-1]
    shadow = Image.new("RGBA", char.size, (0, 0, 0, 0))
    s_layer = Image.new("RGBA", char.size, (0, 0, 0, opacity))
    shadow.paste(s_layer, mask=alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    canvas = Image.new("RGBA", (char.width + abs(offset[0]), char.height + abs(offset[1])), (0, 0, 0, 0))
    canvas.paste(shadow, offset, shadow)
    canvas.paste(char, (0, 0), char)
    return canvas

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width"""
    lines = []
    line = ""
    for word in text.split():
        test_line = line + word + " "
        bbox = font.getbbox(test_line)
        if bbox[2] - bbox[0] <= max_width:
            line = test_line
        else:
            lines.append(line.strip())
            line = word + " "
    if line:
        lines.append(line.strip())
    return lines

def compose_frame(bg: Image.Image, char_img: Image.Image, pos: Tuple[int,int], dialog: Dict, font_name: str, bubble_bg: str, text_color: str) -> Image.Image:
    """Merge character into background with dialogue UI"""
    W, H = bg.size
    frame = bg.convert("RGBA").copy()
    char_clean = remove_white_bg(char_img)
    char_shadowed = add_drop_shadow(char_clean, offset=(12,14), blur=14, opacity=130)
    frame.paste(char_shadowed, pos, char_shadowed)

    draw = ImageDraw.Draw(frame)
    name_font = safe_font(font_name, 42)
    text_font = safe_font(font_name, 44)
    bubble_rgb = tuple(int(bubble_bg.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (200,)
    text_rgb = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    speaker_text = f"{dialog['speaker']} ({dialog['expression']})"
    nb = name_font.getbbox(speaker_text)
    sw, sh = nb[2] - nb[0], nb[3] - nb[1]
    cx = (W - sw) // 2
    draw.rounded_rectangle([cx-24, 24, cx+sw+24, 24+sh+22], radius=14, fill=(0,0,0,180))
    draw.text((cx, 32), speaker_text, font=name_font, fill="#FFE082")

    lines = wrap_text(dialog['text'], text_font, W - 100)
    y = H - 240
    for l in lines:
        tb = text_font.getbbox(l)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        x = (W - tw) // 2
        draw.rounded_rectangle([x-24, y-10, x+tw+24, y+th+12], radius=18, fill=bubble_rgb)
        draw.text((x, y), l, font=text_font, fill=text_rgb)
        y += th + 14
    return frame

# Image Generation
def gen_image_with_retry(url: str, retries: int = 3, backoff_factor: float = 0.5) -> Optional[Image.Image]:
    """Fetch image with retry logic"""
    session = requests.Session()
    retries = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        r = session.get(url, timeout=45)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGBA")
    except Exception as e:
        st.warning(f"Image generation failed: {str(e)}")
        return None

@st.cache_resource(show_spinner="🔧 Warming up character generator…")
def gen_character_expressions(_name: str, _style_prompt: str) -> Dict[str, Image.Image]:
    """Generate character expressions in parallel"""
    expressions = ["neutral", "happy", "sad", "angry", "surprised"]
    out = {}
    style_modifiers = {
        "Modern Anime": "modern anime style, vibrant colors, sharp lines",
        "Vintage Anime": "90s anime style, cel-shaded, grainy texture",
        "Cyberpunk": "cyberpunk anime, neon lights, futuristic",
        "Fantasy": "fantasy anime, magical, elaborate costumes",
        "School Life": "school anime, uniform, bright colors"
    }
    style_modifier = style_modifiers.get(st.session_state.get("char_style_preset", "Modern Anime"), "anime style")

    def fetch_expression(exp):
        prompt = (
            f"{_style_prompt}, {exp} expression, male anime character, standing pose, "
            f"ultra-detailed, {style_modifier}, expressive large eyes, clean edges, high quality, transparent background"
        )
        url = f"https://image.pollinations.ai/prompt/{prompt}"
        img = gen_image_with_retry(url) or Image.new("RGBA", (600, 900), (160, 120, 210, 255))
        return exp, img

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fetch_expression, expressions)
        for exp, img in results:
            out[exp] = img
    return out

@st.cache_resource(show_spinner="🌆 Generating backgrounds…")
def gen_background(_prompt: str, _size=(1080, 1920)) -> Image.Image:
    """Generate background with retry"""
    bg_modifiers = {
        "Cinematic": "cinematic lighting, dramatic composition, film quality",
        "Painterly": "painterly style, brush strokes, artistic",
        "Minimal": "minimalist, clean lines, simple composition",
        "Detailed": "highly detailed, intricate, sharp focus",
        "Moody": "moody atmosphere, dramatic lighting, emotional"
    }
    bg_modifier = bg_modifiers.get(st.session_state.get("bg_style_preset", "Cinematic"), "anime background")
    full_prompt = f"{_prompt}, {bg_modifier}, {st.session_state.get('bg_style_preset', 'Cinematic').lower()} style, high detail"
    url = f"https://image.pollinations.ai/prompt/{full_prompt}"
    img = gen_image_with_retry(url) or Image.new("RGBA", _size, (28, 30, 60, 255))
    return img.resize(_size, Image.LANCZOS)

# Story Processing
def extract_character_names(story: str, limit: int = 3) -> List[str]:
    """Extract character names from story"""
    if len(story.strip()) < 10:
        st.error("Story is too short. Please provide at least one line of dialogue.")
        return []
    if len(story.splitlines()) > 50:
        st.error("Story is too long. Please limit to 50 lines.")
        return []
    
    names = []
    for m in re.finditer(r"^\s*([A-Z][a-zA-Z0-9_]{1,20})\s*:", story, re.M):
        names.append(m.group(1))
    common_words = {"I", "The", "A", "And", "But", "Or", "For", "Nor", "So", "Yet", "With", "At", 
                   "By", "In", "Of", "On", "To", "Up", "As", "It", "Is", "Are", "Was", "Were"}
    sentences = re.split(r'[.!?]+', story)
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) > 1:
            for word in words[1:]:
                if re.match(r'^[A-Z][a-z]{2,}$', word) and word not in common_words:
                    names.append(word)
    seen = set()
    ordered = [n for n in names if not (n in seen or seen.add(n))]
    return ordered[:limit] or ["Akio", "Mira", "Ken"]

def split_story_into_dialogues(story: str, cast: List[str]) -> List[Tuple[str, str, str, str]]:
    """Split story into dialogue tuples"""
    lines = [l.strip() for l in story.splitlines() if l.strip()]
    if not lines:
        lines = [
            f"{cast[0]}: Let's start our adventure!",
            f"{cast[1]}: Whoa, look at that!",
            f"{cast[0]}: Stay focused!",
            f"{cast[2] if len(cast) > 2 else cast[1]}: I have a bad feeling…"
        ]

    def guess_expr(t):
        t_low = t.lower()
        if any(w in t_low for w in ["!", "let's go", "yeah", "awesome", "great", "wonderful", "happy", "love"]):
            return "happy"
        if any(w in t_low for w in ["?", "what", "huh", "wait", "surprise", "shock", "amazing", "wow"]):
            return "surprised"
        if any(w in t_low for w in ["angry", "grr", "mad", "stop", "hate", "damn", "hell", "idiot"]):
            return "angry"
        if any(w in t_low for w in ["sad", "sorry", "alone", "miss", "cry", "tear", "regret", "unfortunate"]):
            return "sad"
        return "neutral"

    dialogues = []
    for i, line in enumerate(lines):
        if ":" in line:
            parts = line.split(":", 1)
            speaker = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            if speaker not in cast:
                speaker = cast[i % len(cast)]
        else:
            speaker = cast[i % len(cast)]
            text = line
        expr = guess_expr(text)
        bg = BG_SUGGESTIONS[i % len(BG_SUGGESTIONS)]
        dialogues.append((speaker, expr, text, bg))
    return dialogues

# TTS
def polly_client():
    """Initialize Polly client"""
    if not (POLLY_AVAILABLE and st.session_state.get("enable_tts", False)):
        return None
    try:
        return boto3.client("polly")
    except Exception:
        return None

def synth_tts_to_temp(text: str, voice_id: str, lang: str) -> Optional[str]:
    """Synthesize TTS to temp file"""
    client = polly_client()
    if client is None:
        return None
    try:
        if len(text) > 300:
            text = text[:300] + "..."
        resp = client.synthesize_speech(
            Text=text, OutputFormat="mp3",
            VoiceId=voice_id, LanguageCode=lang,
            Engine='neural'
        )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(resp["AudioStream"].read())
        tmp.close()
        return tmp.name
    except Exception as e:
        st.warning(f"TTS failed for voice {voice_id}: {str(e)}")
        return None

# Video Building
def add_cam_motion(clip, mode: str):
    """Apply camera motion effect"""
    if mode == "subtle-zoom":
        return clip.resize(lambda t: 1 + 0.02 * np.sin(2 * np.pi * t / 3))
    if mode == "pulse":
        return clip.resize(lambda t: 1 + 0.03 * np.sin(6 * t))
    return clip

def build_video(
    scenes: List[Tuple[str, str, str, str]],
    char_bank: Dict[str, Dict[str, Image.Image]],
    bg_cache: Dict[str, Image.Image],
    per_line_sec: float,
    fps: int,
    motion: str,
    music_file,
    lang: str,
    voices: List[str],
    tts_volume: float,
    music_volume: float
) -> str:
    """Build video from scenes"""
    clips = []
    W, H = CANVAS
    slots = [(W//2-260, H-1100), (60, H-1100), (W-620, H-1100), (W//2-260, H-1150)]
    speak_to_slot = {}
    voice_map = {sp: voices[idx % len(voices)] for idx, sp in enumerate({s for s, _, _, _ in scenes}) if voices}
    audio_segments = []
    total_scenes = len(scenes)
    progress_container = st.empty()
    progress_bar = st.progress(0)

    for idx, (speaker, expr, text, bg_hint) in enumerate(scenes):
        progress_bar.progress((idx + 1) / total_scenes)
        progress_container.text(f"Rendering scene {idx+1}/{total_scenes}: {speaker} - {expr}")
        
        bg = bg_cache.get(bg_hint) or gen_background(bg_hint, size=CANVAS)
        bg_cache[bg_hint] = bg
        if speaker not in speak_to_slot:
            speak_to_slot[speaker] = slots[len(speak_to_slot) % len(slots)]
        pos = speak_to_slot[speaker]
        
        char_img = char_bank.get(speaker, {}).get(expr) or char_bank.get(speaker, {}).get("neutral") or Image.new("RGBA", (600, 900), (150, 120, 200, 255))
        char_img = char_img.resize((520, 820), Image.LANCZOS)
        frame_img = compose_frame(bg, char_img, pos, {
            "speaker": speaker, "expression": expr, "text": text
        }, st.session_state.get("font_choice", "DejaVuSans"), 
        st.session_state.get("bubble_bg", "#14162E"), 
        st.session_state.get("text_color", "#FFFFFF"))
        
        frame_np = np.array(frame_img)
        base = ImageClip(frame_np).set_duration(per_line_sec)
        
        if expr == "angry":
            base = base.fx(vfx.lum_contrast, 0, 35, 160).resize(lambda t: 1 + 0.02 * np.abs(np.sin(12 * t)))
        elif expr == "happy":
            base = base.resize(lambda t: 1 + 0.02 * np.sin(10 * t))
        elif expr == "surprised":
            base = base.resize(lambda t: 1 + 0.04 * np.abs(np.sin(9 * t)))
        elif expr == "sad":
            base = base.fx(vfx.colorx, 0.88)
        
        base = add_cam_motion(base, motion)
        
        line_audio = None
        if st.session_state.get("enable_tts", False) and voice_map.get(speaker):
            tts_path = synth_tts_to_temp(text, voice_map[speaker], lang)
            if tts_path:
                try:
                    line_audio = AudioFileClip(tts_path).volumex(tts_volume)
                except Exception:
                    line_audio = None
                finally:
                    try:
                        os.unlink(tts_path)
                    except:
                        pass
        if line_audio:
            base = base.set_audio(line_audio.set_duration(base.duration))
        
        clips.append(base)
    
    progress_container.empty()
    progress_bar.empty()
    
    if not clips:
        raise RuntimeError("No clips to render")
    
    final = concatenate_videoclips(clips, method="compose", transition=transfx.Crossfade(0.3))
    
    if music_file is not None:
        tmpm = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmpm.write(music_file.read())
        tmpm.close()
        try:
            music_clip = AudioFileClip(tmpm.name).volumex(music_volume)
            final_audio = CompositeAudioClip([music_clip.set_duration(final.duration), final.audio] if final.audio else [music_clip.set_duration(final.duration)])
            final = final.set_audio(final_audio)
        finally:
            try:
                os.unlink(tmpm.name)
            except:
                pass
    
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out.close()
    
    encoding_status = st.empty()
    encoding_status.text("Encoding video...")
    
    try:
        final.write_videofile(
            out.name, fps=fps, codec="libx264", audio_codec="aac", 
            threads=4, verbose=False, logger=None, ffmpeg_params=['-preset', 'fast']
        )
    finally:
        encoding_status.empty()
    
    return out.name

# Main App Workflow
st.title("✨ Anime Shorts Studio — Auto Story → Characters → Video")
st.caption("Paste a short story. I'll pick characters, generate anime shots, add shadows, blend properly, and export a vertical YouTube Short with optional AI voices (Amazon Polly).")

# Sidebar Config
with st.sidebar:
    st.header("🎨 Style & Audio")
    st.session_state.font_choice = st.selectbox("Dialogue Font", ["DejaVuSans", "LiberationSans", "Arial", "AnimeAce"], index=0)
    st.session_state.music_file = st.file_uploader("Background Music (.mp3)", type=["mp3"])
    st.session_state.per_line_duration = st.slider("Seconds per Line", 0.8, 4.0, 1.8, 0.1)
    st.session_state.fps = st.slider("FPS", 18, 30, 24, 1)
    st.session_state.cam_motion = st.selectbox("Camera Motion", ["subtle-zoom", "pulse", "none"], index=0)
    st.session_state.max_chars = st.slider("Max Characters to Detect", 1, 5, 3, 1)
    st.session_state.resolution = st.selectbox("Resolution", ["1080x1920", "720x1280"], index=0)
    CANVAS
    CANVAS = (1080, 1920) if st.session_state.resolution == "1080x1920" else (720, 1280)
    
    st.subheader("🎭 Character Style")
    st.session_state.char_style_preset = st.selectbox("Character Style Preset", 
                                                    ["Modern Anime", "Vintage Anime", "Cyberpunk", "Fantasy", "School Life"], index=0)
    st.subheader("Custom Character Styles")
    custom_styles = st.text_area("Custom styles (JSON)", value=json.dumps(CHAR_STYLE_BANK, indent=2))
    try:
        CHAR_STYLE_BANK = json.loads(custom_styles)
    except:
        st.error("Invalid JSON for custom styles")
    
    st.subheader("🌆 Background Style")
    st.session_state.bg_style_preset = st.selectbox("Background Style Preset", 
                                                  ["Cinematic", "Painterly", "Minimal", "Detailed", "Moody"], index=0)
    
    st.subheader("🗣️ Amazon Polly (Optional)")
    st.session_state.enable_tts = st.toggle("Enable Polly TTS", value=False)
    st.session_state.default_language = st.selectbox("Voice Language", ["en-US", "en-GB", "hi-IN", "ja-JP"], index=0)
    st.session_state.tts_volume = st.slider("TTS Volume", 0.5, 1.5, 1.0, 0.1)
    st.session_state.music_volume = st.slider("Music Volume", 0.0, 0.5, 0.18, 0.01)
    
    st.subheader("🎨 Dialogue Styling")
    st.session_state.bubble_bg = st.color_picker("Dialogue Bubble Color", "#14162E")
    st.session_state.text_color = st.color_picker("Dialogue Text Color", "#FFFFFF")
    
    st.subheader("♿ Accessibility")
    st.session_state.high_contrast = st.toggle("High Contrast Mode")
    if st.session_state.high_contrast:
        st.markdown('<style>.high-contrast { background: #ffffff; color: #000000; }</style>', unsafe_allow_html=True)
    
    st.markdown(f"<span class='badge'>Pillow 10+ · Parallelized · Enhanced UX</span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Progress")
    st.sidebar.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {st.session_state.progress}%"></div></div>', unsafe_allow_html=True)
    st.sidebar.caption(st.session_state.status)

# Main Workflow
st.subheader("📖 Step 1 — Paste Your Short Story")
default_story = """Riku: We finally made it to the rooftop!
Aoi: The view… it's beautiful.
Riku: No time to relax. Something feels off.
Aoi: Wait—did you hear that?
Riku: Stay behind me. We end this tonight!"""
story = st.text_area("Short Story (4–12 lines works best)", default_story, height=200, help="Format like Name: line. If no names given, I'll infer.")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("👥 Step 2 — Auto Detect Cast")
    if st.button("Detect Characters & Plan Scenes"):
        with st.spinner("Analyzing story for characters..."):
            cast = extract_character_names(story, limit=st.session_state.max_chars)
            st.session_state.cast = cast
            st.success(f"Detected Characters: {', '.join(cast)}")
            
            scenes = split_story_into_dialogues(story, cast)
            st.session_state.scenes = scenes
            
            st.write("Planned Scenes:")
            scene_data = [{"scene": i+1, "speaker": s, "expression": e, "background": b, "text": t[:60] + "…" if len(t) > 60 else t}
                          for i, (s, e, t, b) in enumerate(scenes)]
            st.dataframe(scene_data, use_container_width=True)
            
            if st.button("Show Scene Distribution"):
                from collections import Counter
                speakers = [s for s, _, _, _ in scenes]
                speaker_counts = Counter(speakers)
                st.markdown("""```chartjs
                {
                    "type": "bar",
                    "data": {
                        "labels": """ + json.dumps(list(speaker_counts.keys())) + """,
                        "datasets": [{
                            "label": "Lines per Character",
                            "data": """ + json.dumps(list(speaker_counts.values())) + """,
                            "backgroundColor": ["#6e6bff", "#9c7aff", "#ff82b2", "#1fdf64", "#ffca28"],
                            "borderColor": ["#ffffff"],
                            "borderWidth": 1
                        }]
                    },
                    "options": {
                        "scales": {
                            "y": {"beginAtZero": true, "title": {"display": true, "text": "Number of Lines"}},
                            "x": {"title": {"display": true, "text": "Characters"}}
                        }
                    }
                }
                ```""", unsafe_allow_html=True)

with col2:
    st.subheader("🎭 Step 3 — Generate Character Looks")
    st.caption("Each detected character gets a distinct style.")
    if st.button("Generate Character Art"):
        if "cast" not in st.session_state:
            st.warning("Run character detection first.")
        else:
            char_bank = {}
            total_chars = len(st.session_state.cast)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, name in enumerate(st.session_state.cast):
                status_text.text(f"Generating {name}... ({i+1}/{total_chars})")
                progress_bar.progress((i+1) / total_chars)
                style = CHAR_STYLE_BANK[i % len(CHAR_STYLE_BANK)]
                char_bank[name] = gen_character_expressions(name, style)
            
            st.session_state.char_bank = char_bank
            progress_bar.empty()
            status_text.empty()
            st.success("Character expressions ready!")
            st.image([char_bank[n]["neutral"] for n in st.session_state.cast], caption=st.session_state.cast, width=220)

st.subheader("🌆 Step 4 — Prepare Backgrounds")
st.caption("I'll auto-pick backgrounds per line. You can modify in the code bank if you like.")
if st.button("Preload Backgrounds"):
    if "scenes" not in st.session_state:
        st.warning("Detect scenes first.")
    else:
        bg_cache = {}
        unique_bgs = list(set(bg for _, _, _, bg in st.session_state.scenes))
        total_bgs = len(unique_bgs)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, bg in enumerate(unique_bgs):
            status_text.text(f"Generating {bg}... ({i+1}/{total_bgs})")
            progress_bar.progress((i+1) / total_bgs)
            if bg not in bg_cache:
                bg_cache[bg] = gen_background(bg, size=CANVAS)
        
        st.session_state.bg_cache = bg_cache
        progress_bar.empty()
        status_text.empty()
        st.success(f"Loaded {len(bg_cache)} backgrounds.")
        st.image(list(bg_cache.values())[:3], caption=list(bg_cache.keys())[:3], width=240)

st.subheader("🎬 Step 5 — Render Anime Short")
if st.button("Preview Single Frame"):
    if "scenes" in st.session_state and "char_bank" in st.session_state:
        scene = st.session_state.scenes[0]
        bg = st.session_state.get("bg_cache", {}).get(scene[3]) or gen_background(scene[3], size=CANVAS)
        frame = compose_frame(bg, st.session_state.char_bank[scene[0]][scene[1]], 
                             (CANVAS[0]//2-260, CANVAS[1]-1100), 
                             {"speaker": scene[0], "expression": scene[1], "text": scene[2]},
                             st.session_state.get("font_choice", "DejaVuSans"),
                             st.session_state.get("bubble_bg", "#14162E"),
                             st.session_state.get("text_color", "#FFFFFF"))
        st.image(frame, caption="Frame Preview")

if st.button("🚀 Generate Video"):
    if "scenes" not in st.session_state:
        st.error("Please detect characters & plan scenes first.")
    elif "char_bank" not in st.session_state:
        st.error("Please generate character art first.")
    else:
        try:
            with st.spinner("Rendering…"):
                path = build_video(
                    scenes=st.session_state.scenes,
                    char_bank=st.session_state.char_bank,
                    bg_cache=st.session_state.get("bg_cache", {}),
                    per_line_sec=st.session_state.per_line_duration,
                    fps=st.session_state.fps,
                    motion=st.session_state.cam_motion,
                    music_file=st.session_state.get("music_file"),
                    lang=st.session_state.default_language,
                    voices=VOICE_POOLS.get(st.session_state.default_language, []),
                    tts_volume=st.session_state.tts_volume,
                    music_volume=st.session_state.music_volume
                )
            st.success("✅ Video ready!")
            st.video(path)
            video_duration = len(st.session_state.scenes) * st.session_state.per_line_duration
            st.info(f"Video duration: {video_duration:.1f} seconds, {len(st.session_state.scenes)} scenes")
            
            with open(path, "rb") as f:
                st.download_button("⬇️ Download MP4", f.read(), file_name="anime_short.mp4", mime="video/mp4", help="Download the generated video")
            
            try:
                os.unlink(path)
            except:
                pass
        except Exception as e:
            st.error(f"Error rendering video: {str(e)}")
            st.exception(e)
"""Local recording booth for the fuzzy-phoneme pilot dataset.

A small Streamlit app that walks through the pilot word list, records
each word via the browser mic (works reliably in Safari because a
localhost Streamlit server is a secure context, unlike an embedded
iframe), and saves clips directly into Proposal/clips/<pair_id>/ using
the project's filename convention.

Usage:
    pip install -r Proposal/scripts/requirements.txt
    streamlit run Proposal/scripts/recording_booth.py

Then open the local URL it prints (usually http://localhost:8501) in
Safari/Chrome and record. Files land directly in Proposal/clips/.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_mic_recorder import mic_recorder

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = REPO_ROOT / "Proposal" / "clips"

# pair_id -> (display name, [(romanized, devanagari, label, meaning), ...])
PAIRS: dict[str, tuple[str, list[tuple[str, str, str, str]]]] = {
    "ba_bha": ("ब / भ — aspirated vs unaspirated", [
        ("baani", "बानी", "ba", "habit"), ("bhaani", "भानी", "bha", "—"),
        ("bar", "बर", "ba", "—"), ("bhar", "भर", "bha", "full"),
        ("baag", "बाग", "ba", "tiger"), ("bhaag", "भाग", "bha", "part"),
        ("baala", "बाला", "ba", "—"), ("bhaala", "भाला", "bha", "spear"),
        ("boot", "बूट", "ba", "boot"), ("bhoot", "भूत", "bha", "ghost"),
    ]),
    "ka_kha": ("क / ख — aspirated vs unaspirated", [
        ("kaam", "काम", "ka", "work"), ("khaam", "खाम", "kha", "envelope"),
        ("kaaro", "कारो", "ka", "—"), ("khaaro", "खारो", "kha", "—"),
        ("kaali", "काली", "ka", "black"), ("khaali", "खाली", "kha", "empty"),
        ("kaan", "कान", "ka", "ear"), ("khaan", "खान", "kha", "mine"),
        ("kuwa", "कुवा", "ka", "well"), ("khuwa", "खुवा", "kha", "—"),
    ]),
    "tta_ta": ("ट / त — retroflex vs dental", [
        ("ttaaro", "टारो", "tta", "—"), ("taaro", "तारो", "ta", "target"),
        ("ttaauko", "टाउको", "tta", "head"), ("taauko", "ताउको", "ta", "—"),
        ("ttikaa", "टीका", "tta", "tika"), ("tikaa", "तीका", "ta", "—"),
        ("ttol", "टोल", "tta", "neighborhood"), ("tol", "तोल", "ta", "weight"),
        ("ttaar", "टार", "tta", "—"), ("taar", "तार", "ta", "wire"),
    ]),
    "dda_da": ("ड / द — retroflex vs dental", [
        ("ddaal", "डाल", "dda", "branch"), ("daal", "दाल", "da", "lentils"),
        ("ddori", "डोरी", "dda", "rope"), ("dori", "दोरी", "da", "—"),
        ("ddaanda", "डाँडा", "dda", "hill"), ("daant", "दाँत", "da", "tooth"),
        ("ddar", "डर", "dda", "fear"), ("dar", "दर", "da", "rate"),
        ("ddaak", "डाक", "dda", "post"), ("daag", "दाग", "da", "stain"),
    ]),
    "i_ii": ("इ / ई — vowel length", [
        ("din", "दिन", "i", "day"), ("deen", "दीन", "ii", "poor"),
        ("pir", "पिर", "i", "worry"), ("peer", "पीर", "ii", "—"),
        ("kin", "किन", "i", "why"), ("keen", "कीन", "ii", "—"),
        ("sima", "सिमा", "i", "—"), ("seemaa", "सीमा", "ii", "border"),
        ("mil", "मिल", "i", "—"), ("meel", "मील", "ii", "mile"),
    ]),
    "u_uu": ("उ / ऊ — vowel length", [
        ("mul", "मुल", "u", "main"), ("mool", "मूल", "uu", "source"),
        ("pul", "पुल", "u", "bridge"), ("pool", "पूल", "uu", "—"),
        ("sut", "सुत", "u", "—"), ("soot", "सूत", "uu", "thread"),
        ("bhul", "भुल", "u", "—"), ("bhool", "भूल", "uu", "mistake"),
        ("dudh", "दुध", "u", "milk"), ("doodh", "दूध", "uu", "—"),
    ]),
    "sha_sa": ("श / स — sibilant merger", [
        ("shaal", "शाल", "sha", "shawl"), ("saal", "साल", "sa", "year"),
        ("shar", "शर", "sha", "—"), ("sar", "सर", "sa", "sir"),
        ("shakti", "शक्ति", "sha", "power"), ("sakti", "सक्ति", "sa", "—"),
        ("aashaa", "आशा", "sha", "hope"), ("aasaa", "आसा", "sa", "—"),
        ("shahar", "शहर", "sha", "city"), ("sahar", "सहर", "sa", "—"),
    ]),
    "ssha_sa": ("ष / स — sibilant merger", [
        ("ssat", "षट्", "ssha", "—"), ("sat", "सट्", "sa", "—"),
        ("bhaassaa", "भाषा", "ssha", "language"), ("bhaasaa", "भासा", "sa", "—"),
        ("doss", "दोष", "ssha", "fault"), ("dos", "दोस", "sa", "—"),
        ("koss", "कोष", "ssha", "fund"), ("kos", "कोस", "sa", "a distance unit"),
        ("bissay", "विषय", "ssha", "subject"), ("visay", "विसय", "sa", "—"),
    ]),
}

ITEMS = [
    {"pair_id": pid, "pair_name": name, "romanized": rom, "devanagari": dev,
     "label": label, "meaning": meaning}
    for pid, (name, words) in PAIRS.items()
    for (rom, dev, label, meaning) in words
]

st.set_page_config(page_title="Phoneme Recording Booth", page_icon="🎙️", layout="centered")

if "idx" not in st.session_state:
    st.session_state.idx = 0

speaker = st.sidebar.text_input("Speaker ID", value=st.session_state.get("speaker", ""),
                                 placeholder="e.g. spk02").strip()
st.session_state.speaker = speaker


def clip_path(item: dict) -> Path:
    sp = speaker or "spkXX"
    fname = f"{sp}_{item['romanized']}_{item['label']}.wav"
    return CLIPS_DIR / item["pair_id"] / fname


def recorded_set() -> set[str]:
    done = set()
    for item in ITEMS:
        if clip_path(item).exists():
            done.add(item["pair_id"] + item["romanized"])
    return done


done = recorded_set()

st.sidebar.progress(len(done) / len(ITEMS) if ITEMS else 0)
st.sidebar.caption(f"{len(done)} of {len(ITEMS)} recorded")

st.sidebar.markdown("---")
for pid, (name, words) in PAIRS.items():
    st.sidebar.markdown(f"**{name}**")
    for rom, dev, label, meaning in words:
        key = pid + rom
        mark = "✅" if key in done else "⬜️"
        idx = next(i for i, it in enumerate(ITEMS) if it["pair_id"] == pid and it["romanized"] == rom)
        if st.sidebar.button(f"{mark} {dev}  ·  {rom}", key=f"jump_{key}"):
            st.session_state.idx = idx
            st.rerun()

st.session_state.idx = max(0, min(st.session_state.idx, len(ITEMS) - 1))
item = ITEMS[st.session_state.idx]

st.title("🎙️ Phoneme Recording Booth")
st.caption("Nepali Fuzzy Phonemes — pilot clip collection (local, saves straight into Proposal/clips/)")

if not speaker:
    st.warning("Enter a speaker ID in the sidebar before recording.")

st.markdown(f"### {item['pair_name']}")
st.markdown(
    f"<div style='font-size:4rem; font-weight:700; margin:0.2rem 0;'>{item['devanagari']}</div>",
    unsafe_allow_html=True,
)
meaning = f" — {item['meaning']}" if item["meaning"] != "—" else ""
st.markdown(f"**{item['romanized']}**{meaning}")
target_path = clip_path(item)
st.code(str(target_path.relative_to(REPO_ROOT)), language=None)

audio = mic_recorder(start_prompt="🔴 Record", stop_prompt="⏹ Stop", key=f"rec_{st.session_state.idx}")

if audio:
    st.audio(audio["bytes"])
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Keep & continue", disabled=not speaker, use_container_width=True):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(audio["bytes"])
            nxt = next((i for i, it in enumerate(ITEMS)
                        if (it["pair_id"] + it["romanized"]) not in recorded_set()), None)
            st.session_state.idx = nxt if nxt is not None else st.session_state.idx
            st.success(f"Saved {target_path.name}")
            st.rerun()
    with col2:
        if st.button("↺ Redo", use_container_width=True):
            st.rerun()

st.markdown("---")
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("← Previous word", use_container_width=True, disabled=st.session_state.idx == 0):
        st.session_state.idx -= 1
        st.rerun()
with col_next:
    if st.button("Next word →", use_container_width=True, disabled=st.session_state.idx >= len(ITEMS) - 1):
        st.session_state.idx += 1
        st.rerun()

if len(done) == len(ITEMS):
    st.balloons()
    st.success("All words recorded for this speaker!")

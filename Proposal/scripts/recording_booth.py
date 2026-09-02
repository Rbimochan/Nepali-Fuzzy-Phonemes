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
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder

# streamlit-mic-recorder always captures from the OS/browser's default input
# device (no per-recording device selection). This widget is a monitor only —
# it never records or saves anything — so you can confirm a specific mic
# (e.g. a wireless lavalier) is actually picking up your voice before
# recording for real below. It auto-selects any device whose name suggests
# a wireless/lavalier mic. To make the recorder below actually use that
# device, set it as the *system* default input (macOS: System Settings >
# Sound > Input) and refresh the page — the picker here can preview any
# device, but capture always follows the OS default.
#
# IMPORTANT: this monitor only opens the mic stream while explicitly
# "testing" (Test/Stop button), and always releases it afterwards. Some
# devices — a wireless lavalier's USB/Bluetooth receiver especially — only
# allow one open capture session at a time, so an always-on preview stream
# would fight the recorder below for the device and make Record silently
# fail. Always press Stop test before recording for real.
MIC_PICKER_HTML = """
<div id="mic-picker" style="font-family:-apple-system,sans-serif;">
  <select id="mic-select" style="width:100%;padding:0.5rem;border-radius:6px;
    border:1px solid #444;background:#1b1f21;color:#eee;font-size:0.85rem;">
    <option>Requesting microphone permission…</option>
  </select>

  <div style="margin-top:0.6rem;border:1px solid #2c3230;border-radius:8px;
    background:#0f1213;padding:0.4rem;">
    <canvas id="mic-wave" style="width:100%;height:70px;display:block;"></canvas>
  </div>

  <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;">
    <span style="font-size:0.7rem;color:#9aa39c;width:2.2rem;">peak</span>
    <div style="flex:1;height:8px;border-radius:999px;background:#20241f;overflow:hidden;">
      <div id="mic-level" style="height:100%;width:0%;background:#6bc79a;transition:width 0.05s linear;"></div>
    </div>
    <button id="mic-toggle" style="padding:0.35rem 0.8rem;border-radius:999px;border:1px solid #6bc79a;
      background:transparent;color:#6bc79a;font-size:0.78rem;font-weight:600;cursor:pointer;white-space:nowrap;">
      ▶ Test mic
    </button>
  </div>

  <div id="mic-status" style="margin-top:0.4rem;font-size:0.78rem;color:#9aa39c;">
    Monitor is off — mic isn't in use. Press "Test mic" to check it, "Stop test" before recording for real.
  </div>
</div>
<script>
(async function () {
  const select = document.getElementById("mic-select");
  const levelBar = document.getElementById("mic-level");
  const status = document.getElementById("mic-status");
  const canvas = document.getElementById("mic-wave");
  const toggleBtn = document.getElementById("mic-toggle");
  const ctx2d = canvas.getContext("2d");
  let currentStream = null;
  let rafId = null;
  let audioCtx = null;
  let monitoring = false;

  function stopMonitor(reason) {
    monitoring = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    if (currentStream) { currentStream.getTracks().forEach(t => t.stop()); currentStream = null; }
    if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
    ctx2d.clearRect(0, 0, canvas.width, canvas.height);
    levelBar.style.width = "0%";
    toggleBtn.textContent = "▶ Test mic";
    if (reason) status.textContent = reason;
  }

  function looksLikeLavalier(label) {
    return /lavalier|lapel|wireless|lav\\b/i.test(label);
  }

  async function listDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter(d => d.kind === "audioinput");
      select.innerHTML = "";
      if (!inputs.length) {
        select.innerHTML = "<option>No microphones found</option>";
        return;
      }
      inputs.forEach((d, i) => {
        const opt = document.createElement("option");
        opt.value = d.deviceId;
        opt.textContent = d.label || ("Microphone " + (i + 1));
        select.appendChild(opt);
      });
      const lav = inputs.find(d => looksLikeLavalier(d.label));
      if (lav) {
        select.value = lav.deviceId;
        status.textContent = "Auto-selected \\"" + (lav.label || "wireless mic") + "\\" — looks like your lavalier. Press \\"Test mic\\" to check it.";
      }
    } catch (e) {
      select.innerHTML = "<option>Could not list devices</option>";
      status.textContent = String(e);
    }
  }

  async function startMonitor() {
    try {
      const deviceId = select.value;
      const constraints = { audio: deviceId ? { deviceId: { exact: deviceId } } : true };
      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(currentStream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      const timeData = new Uint8Array(analyser.frequencyBinCount);

      canvas.width = canvas.clientWidth * devicePixelRatio;
      canvas.height = canvas.clientHeight * devicePixelRatio;

      monitoring = true;
      toggleBtn.textContent = "■ Stop test";
      status.textContent = "Testing — nothing is recorded or saved. Speak into the mic.";

      (function loop() {
        if (!monitoring) return;
        rafId = requestAnimationFrame(loop);
        analyser.getByteTimeDomainData(timeData);

        const w = canvas.width, h = canvas.height;
        ctx2d.clearRect(0, 0, w, h);
        ctx2d.strokeStyle = "#6bc79a";
        ctx2d.lineWidth = 2 * devicePixelRatio;
        ctx2d.beginPath();
        const sliceWidth = w / timeData.length;
        let x = 0;
        let sumSq = 0;
        for (let i = 0; i < timeData.length; i++) {
          const v = (timeData[i] - 128) / 128;
          sumSq += v * v;
          const y = (v * 0.5 + 0.5) * h;
          if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
          x += sliceWidth;
        }
        ctx2d.stroke();

        const rms = Math.sqrt(sumSq / timeData.length);
        levelBar.style.width = Math.min(100, rms * 300) + "%";
      })();
    } catch (e) {
      status.textContent = "Mic test unavailable: " + e.message;
    }
  }

  toggleBtn.addEventListener("click", () => {
    if (monitoring) {
      stopMonitor("Stopped — mic is free for recording.");
    } else {
      startMonitor();
    }
  });

  select.addEventListener("change", () => {
    if (monitoring) startMonitor(); // restart on the newly chosen device
  });

  // Safety net: never leave the mic open if the tab/frame goes away or hides.
  window.addEventListener("beforeunload", () => stopMonitor());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopMonitor("Stopped (tab hidden) — mic is free.");
  });

  try {
    // Trigger a brief permission prompt so device labels are populated,
    // then immediately release it — this does not start monitoring.
    const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
    tmp.getTracks().forEach(t => t.stop());
  } catch (e) {
    status.textContent = "Microphone permission needed to list devices.";
  }
  await listDevices();
  navigator.mediaDevices.ondevicechange = listDevices;
})();
</script>
"""

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

with st.expander("🎚️ Mic input monitor", expanded=True):
    components.html(MIC_PICKER_HTML, height=200)

# A redo counter per word gives the recorder widget a fresh key each time,
# so "Redo" fully remounts it (new mic session) instead of just re-showing
# the same cached take — st.rerun() alone doesn't clear a component's
# last returned value.
if "redo_counts" not in st.session_state:
    st.session_state.redo_counts = {}
redo_n = st.session_state.redo_counts.get(st.session_state.idx, 0)

audio = mic_recorder(
    start_prompt="🔴 Record", stop_prompt="⏹ Stop",
    key=f"rec_{st.session_state.idx}_{redo_n}",
)

if audio:
    st.audio(audio["bytes"])
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Keep & continue", disabled=not speaker, use_container_width=True):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(audio["bytes"])
            st.session_state.redo_counts.pop(st.session_state.idx, None)
            nxt = next((i for i, it in enumerate(ITEMS)
                        if (it["pair_id"] + it["romanized"]) not in recorded_set()), None)
            st.session_state.idx = nxt if nxt is not None else st.session_state.idx
            st.success(f"Saved {target_path.name}")
            st.rerun()
    with col2:
        if st.button("↺ Redo (restart recorder)", use_container_width=True):
            st.session_state.redo_counts[st.session_state.idx] = redo_n + 1
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

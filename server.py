# Copyright 2026 Sanghyun Jo. Licensed under Apache 2.0.
# AI Image Quality Boost Battle: FastAPI backend with multi-GPU editing and leaderboard.

import os
import io
import json
import uuid
import time
import asyncio
import base64
import logging
import glob
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("boost-battle")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
DEVICE_IDS = list(range(len(VISIBLE_DEVICES.split(","))))
SCORER_DEVICE = DEVICE_IDS[0]       # First GPU for scoring models
EDITOR_DEVICES = DEVICE_IDS[1:]     # Remaining GPUs for ICEdit instances

IMAGE_SIZE = 512
GPU_IDLE_TIMEOUT = 60               # Auto-release GPU lock after N seconds idle
LEADERBOARD_DIR = os.environ.get("LEADERBOARD_DIR", "./leaderboard")

# Prompt used by HPSv3 for visual quality scoring
GENERIC_PROMPT = (
    "A visually stunning, high-resolution artwork with excellent composition, "
    "vibrant colors, detailed textures, balanced lighting, and professional quality."
)

LORA_WEIGHTS = os.environ.get("LORA_WEIGHTS", "./weights/ICEdit-MoE-LoRA.safetensors")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

scorer = None
editors = {}


def load_models():
    """Load scoring models (HPSv3 + MOAT) and ICEdit editing pipelines."""
    global scorer, editors
    from core.hpsv3 import HPSv3PytorchRunner
    from core.moat import DanbooruMOAT

    logger.info(f"Loading scorer on cuda:{SCORER_DEVICE}")
    hpsv3 = HPSv3PytorchRunner(device=f"cuda:{SCORER_DEVICE}")
    moat = DanbooruMOAT("./weights/DanbooruMOAT.onnx")
    scorer = {"hpsv3": hpsv3, "moat": moat}

    if not EDITOR_DEVICES:
        logger.warning("No editor GPUs available. Editing will be disabled.")
        return

    import sys
    import types
    import torch
    import torch.nn as nn
    import sanghyunjo as shjo

    # Stub for transformers.modeling_layers (required by peft_icedit, absent in transformers 4.48.3)
    if "transformers.modeling_layers" not in sys.modules:
        _stub = types.ModuleType("transformers.modeling_layers")
        class _GCLayer(nn.Module):
            pass
        _stub.GradientCheckpointingLayer = _GCLayer
        sys.modules["transformers.modeling_layers"] = _stub

    # Replace peft with MoE-LoRA-compatible local version
    local_peft_src = os.path.abspath(
        ("/mnt/nas5/" if shjo.linux() else "//192.168.100.192/Data/") + "peft_icedit/src"
    )
    for k in list(sys.modules.keys()):
        if k.startswith("peft"):
            del sys.modules[k]
    sys.path.insert(0, local_peft_src)

    import diffusers.utils.peft_utils as peft_utils
    from core import diffusion
    from demo_inp_icedit import DitInpaintPipeline, get_peft_kwargs

    peft_utils.get_peft_kwargs = get_peft_kwargs

    for dev in EDITOR_DEVICES:
        logger.info(f"Loading ICEdit on cuda:{dev}")
        pipe = diffusion.build_pipeline("FLUX.1-Fill-dev", device=torch.device(f"cuda:{dev}"))
        pipe.load_lora_weights(LORA_WEIGHTS)
        editors[dev] = DitInpaintPipeline(pipe)

    logger.info(f"Ready: scorer=cuda:{SCORER_DEVICE}, editors=cuda:{EDITOR_DEVICES}")


# ---------------------------------------------------------------------------
# GPU Pool (async lock per editor GPU with idle auto-release)
# ---------------------------------------------------------------------------

class GPUPool:
    def __init__(self, device_ids: list[int]):
        self.locks = {d: asyncio.Lock() for d in device_ids}
        self.last_active: dict[int, float] = {d: 0.0 for d in device_ids}
        self.holders: dict[int, str | None] = {d: None for d in device_ids}

    async def status(self) -> dict:
        now = time.time()
        available, busy = [], []
        for d in self.locks:
            if not self.locks[d].locked():
                available.append(d)
            else:
                elapsed = now - self.last_active[d]
                remaining = max(0, GPU_IDLE_TIMEOUT - elapsed)
                busy.append({"gpu": d, "holder": self.holders[d],
                             "idle_sec": round(elapsed, 1),
                             "auto_release_in": round(remaining, 1)})
        shortest_wait = min((b["auto_release_in"] for b in busy), default=0.0)
        return {
            "available_count": len(available), "total": len(self.locks),
            "available_gpus": available, "busy_gpus": busy,
            "shortest_wait_sec": round(shortest_wait, 1),
        }

    async def acquire(self, session_id: str) -> int | None:
        for d, lock in self.locks.items():
            if not lock.locked():
                await lock.acquire()
                self.last_active[d] = time.time()
                self.holders[d] = session_id
                return d
        return None

    def touch(self, device_id: int):
        self.last_active[device_id] = time.time()

    def release(self, device_id: int):
        if device_id in self.locks and self.locks[device_id].locked():
            self.locks[device_id].release()
            self.holders[device_id] = None

    async def auto_release_loop(self):
        while True:
            await asyncio.sleep(5)
            now = time.time()
            for d in list(self.locks.keys()):
                if self.locks[d].locked() and (now - self.last_active[d]) > GPU_IDLE_TIMEOUT:
                    logger.info(f"Auto-releasing idle GPU cuda:{d} (holder={self.holders[d]})")
                    self.release(d)


gpu_pool = GPUPool(EDITOR_DEVICES)

# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def preprocess(image: Image.Image) -> Image.Image:
    """Apply EXIF rotation, convert to RGB, resize shorter side to IMAGE_SIZE, center-crop to square."""
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    image = image.convert("RGB")
    w, h = image.size
    if w < h:
        new_w, new_h = IMAGE_SIZE, int(h * (IMAGE_SIZE / w))
    else:
        new_h, new_w = IMAGE_SIZE, int(w * (IMAGE_SIZE / h))
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - IMAGE_SIZE) // 2, (new_h - IMAGE_SIZE) // 2
    return image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))


def compute_score(image: Image.Image) -> dict:
    """Score visual quality using HPSv3. Returns {score, logvar}."""
    [mu, logvar] = scorer["hpsv3"].predict([image], [GENERIC_PROMPT])[0]
    return {"score": round(float(mu), 3), "logvar": round(float(logvar), 3)}


def check_tags(image: Image.Image) -> dict:
    """Run MOAT tagger to get content rating and general tags."""
    rating, general_tags, _ = scorer["moat"].predict(image)
    nsfw = float(rating.get("explicit", 0)) + float(rating.get("questionable", 0))
    top_tags = sorted(general_tags.items(), key=lambda x: x[1], reverse=True)[:8]
    return {
        "tags": [{"name": t[0], "score": round(float(t[1]), 3)} for t in top_tags],
        "rating": {k: round(float(v), 3) for k, v in rating.items()},
        "nsfw_score": round(nsfw, 3),
    }


def run_icedit(device_id: int, image: Image.Image, prompt: str, seed: int = 0) -> Image.Image:
    """Run ICEdit on a single image and return the edited PIL image."""
    import torch
    import torchvision.transforms.functional as TF
    import sanghyunjo.ai_utils as shai

    pipe = editors[device_id]
    img = image.convert("RGB").resize((pipe.image_size, pipe.image_size))
    tensor = TF.to_tensor(img) * 2 - 1
    result_bgr = pipe.edit(
        tensor, prompt,
        steps=28, cfg=50.0,
        generator=shai.set_seed(seed),
    )
    return Image.fromarray(result_bgr[:, :, ::-1])


def encode_img(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def encode_img_from_path(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Leaderboard persistence (folder-based)
#
# Structure:
#   leaderboard/
#   ├── _ranking.json              <- auto-generated summary for review
#   ├── {session_id}/
#   │   ├── session.json           <- metadata (creator, scores, timestamps)
#   │   ├── original.png
#   │   ├── best.png               <- best edited image
#   │   ├── round_01/
#   │   │   ├── edited.png
#   │   │   └── info.json          <- {prompt, score, delta, timestamp}
#   │   └── ...
#
# Management:
#   - Remove a bad entry:  rm -rf leaderboard/{session_id}/
#   - Reset everything:    rm -rf leaderboard/*/
#   - Review rankings:     cat leaderboard/_ranking.json
# ---------------------------------------------------------------------------

os.makedirs(LEADERBOARD_DIR, exist_ok=True)


def _session_dir(sid: str) -> str:
    return os.path.join(LEADERBOARD_DIR, sid)


def _save_session_init(sid: str, creator: str, original: Image.Image, initial_score: float):
    sdir = _session_dir(sid)
    os.makedirs(sdir, exist_ok=True)
    original.save(os.path.join(sdir, "original.png"))
    meta = {
        "session_id": sid, "creator": creator,
        "initial_score": initial_score, "best_score": initial_score,
        "best_round": None, "gap": 0.0,
        "created_at": datetime.now(KST).isoformat(), "total_rounds": 0,
    }
    with open(os.path.join(sdir, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _save_round(sid: str, round_num: int,
                prompt: str, edited: Image.Image, score: float, delta: float):
    rdir = os.path.join(_session_dir(sid), f"round_{round_num:02d}")
    os.makedirs(rdir, exist_ok=True)
    edited.save(os.path.join(rdir, "edited.png"))
    with open(os.path.join(rdir, "info.json"), "w") as f:
        json.dump({"round": round_num, "prompt": prompt, "score": score,
                    "delta": delta, "timestamp": datetime.now(KST).isoformat()}, f, indent=2)


def _update_session_best(sid: str, best_score: float, best_round: int,
                         gap: float, edited: Image.Image, total_rounds: int):
    sdir = _session_dir(sid)
    edited.save(os.path.join(sdir, "best.png"))
    meta_path = os.path.join(sdir, "session.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"best_score": best_score, "best_round": best_round, "gap": gap,
                 "total_rounds": total_rounds, "updated_at": datetime.now(KST).isoformat()})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    _rebuild_ranking()


def _update_session_rounds(sid: str, total_rounds: int):
    meta_path = os.path.join(_session_dir(sid), "session.json")
    if not os.path.exists(meta_path):
        return
    meta = json.load(open(meta_path))
    meta["total_rounds"] = total_rounds
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _rebuild_ranking():
    """Scan session folders and regenerate _ranking.json."""
    entries = []
    for path in sorted(glob.glob(os.path.join(LEADERBOARD_DIR, "*", "session.json"))):
        try:
            meta = json.load(open(path))
            entries.append({
                "folder": os.path.basename(os.path.dirname(path)),
                "session_id": meta.get("session_id", ""),
                "creator": meta.get("creator", ""),
                "initial_score": meta.get("initial_score", 0),
                "best_score": meta.get("best_score", 0),
                "gap": meta.get("gap", 0),
                "best_round": meta.get("best_round"),
                "total_rounds": meta.get("total_rounds", 0),
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", meta.get("created_at", "")),
            })
        except Exception:
            pass
    entries.sort(key=lambda x: x["gap"], reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    with open(os.path.join(LEADERBOARD_DIR, "_ranking.json"), "w") as f:
        json.dump(entries, f, indent=2)
    return entries


def _load_leaderboard_for_web(limit: int = 3) -> list[dict]:
    """Load top leaderboard entries with base64 images for the web API."""
    ranking_path = os.path.join(LEADERBOARD_DIR, "_ranking.json")
    if not os.path.exists(ranking_path):
        entries = _rebuild_ranking()
    else:
        entries = json.load(open(ranking_path))

    # Validate folders still exist; rebuild if any were deleted
    valid = [e for e in entries
             if os.path.isdir(os.path.join(LEADERBOARD_DIR, e["folder"])) and e.get("gap", 0) > 0]
    if len(valid) != len(entries):
        entries = _rebuild_ranking()
        valid = [e for e in entries
                 if os.path.isdir(os.path.join(LEADERBOARD_DIR, e["folder"])) and e.get("gap", 0) > 0]

    result = []
    for e in valid[:limit]:
        sdir = os.path.join(LEADERBOARD_DIR, e["folder"])
        orig_path, best_path = os.path.join(sdir, "original.png"), os.path.join(sdir, "best.png")
        if not os.path.exists(orig_path) or not os.path.exists(best_path):
            continue
        result.append({
            "session_id": e["session_id"], "creator": e["creator"],
            "initial_score": e["initial_score"], "final_score": e["best_score"],
            "gap": e["gap"], "updated_at": e.get("updated_at", ""),
            "original_image": encode_img_from_path(orig_path),
            "final_image": encode_img_from_path(best_path),
        })
    return result


# ---------------------------------------------------------------------------
# App state and FastAPI setup
# ---------------------------------------------------------------------------

sessions: dict[str, dict] = {}

app = FastAPI(title="AI Image Quality Boost Battle")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    load_models()
    _rebuild_ranking()
    asyncio.create_task(gpu_pool.auto_release_loop())


@app.get("/api/gpu/status")
async def api_gpu_status():
    return await gpu_pool.status()


@app.get("/api/leaderboard")
async def api_leaderboard():
    return _load_leaderboard_for_web(limit=3)


@app.post("/api/session/create")
async def api_create_session(image: UploadFile, creator_name: str = Form("anonymous")):
    """Upload an image and start a new editing session."""
    img = preprocess(Image.open(image.file))
    tags_info = check_tags(img)
    score = compute_score(img)
    sid = str(uuid.uuid4())[:8]
    creator = creator_name.strip()[:20] or "anonymous"

    original_top5 = {t["name"] for t in tags_info["tags"][:5]}
    sessions[sid] = {
        "creator": creator, "original": img,
        "initial_score": score["score"], "best_score": score["score"],
        "best_image": None, "turn": 0,
        "original_tags": tags_info["tags"], "original_top5": original_top5,
    }
    _save_session_init(sid, creator, img, score["score"])

    return {
        "session_id": sid, "creator": creator,
        "score": score["score"], "logvar": score["logvar"],
        "tags": tags_info["tags"], "rating": tags_info["rating"],
        "nsfw_score": tags_info["nsfw_score"],
        "original_image": encode_img(img),
    }


@app.post("/api/session/{sid}/edit")
async def api_edit(sid: str, editing_prompt: str = Form(...)):
    """Edit the original image with a text prompt and score the result."""
    state = sessions.get(sid)
    if not state:
        raise HTTPException(404, "Session not found.")
    if not EDITOR_DEVICES:
        raise HTTPException(503, "No editor GPUs configured.")

    device = await gpu_pool.acquire(sid)
    if device is None:
        status = await gpu_pool.status()
        return {"error": "gpu_busy",
                "message": f"All GPUs busy. Next available in ~{status['shortest_wait_sec']}s",
                "shortest_wait_sec": status["shortest_wait_sec"]}

    try:
        edited = run_icedit(device, state["original"], editing_prompt)
        score = compute_score(edited)
        edited_tags_info = check_tags(edited)
        gpu_pool.touch(device)
    finally:
        gpu_pool.release(device)

    state["turn"] += 1
    turn_id = state["turn"]
    delta = round(score["score"] - state["initial_score"], 3)

    # Tag overlap: edited top-5 must share at least 1 tag with original top-5
    edited_top5 = {t["name"] for t in edited_tags_info["tags"][:5]}
    tag_overlap = state["original_top5"] & edited_top5
    tags_valid = len(tag_overlap) > 0

    logger.info(
        f"[{sid}/@{state['creator']}] Round {turn_id}: "
        f"score={score['score']} delta={delta} tags_valid={tags_valid} "
        f"overlap={tag_overlap} prompt=\"{editing_prompt}\""
    )

    _save_round(sid, turn_id, editing_prompt, edited, score["score"], delta)

    # Update leaderboard only if tag check passes
    if tags_valid and score["score"] > state["best_score"]:
        state["best_score"] = score["score"]
        state["best_image"] = edited
        gap = round(state["best_score"] - state["initial_score"], 3)
        _update_session_best(sid, state["best_score"], turn_id, gap, edited, state["turn"])
    else:
        _update_session_rounds(sid, state["turn"])

    return {
        "turn_id": turn_id, "score": score["score"], "delta": delta,
        "image": encode_img(edited),
        "best_score": state["best_score"],
        "best_gap": round(state["best_score"] - state["initial_score"], 3),
        "edited_tags": edited_tags_info["tags"],
        "edited_rating": edited_tags_info["rating"],
        "tags_valid": tags_valid,
        "original_top5": list(state["original_top5"]),
    }


@app.post("/api/session/{sid}/reset")
async def api_reset(sid: str):
    """End the current session and return to the upload screen."""
    sessions.pop(sid, None)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Static file serving (with cache busting for index.html)
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    with open("web/index.html", "r") as f:
        html = f.read()
    return HTMLResponse(content=html, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache", "Expires": "0",
    })

app.mount("/web", StaticFiles(directory="web", html=False), name="web")

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--tunnel", action="store_true", help="Open a public ngrok tunnel")
    args = parser.parse_args()

    port = int(os.environ.get("PORT", 7960))
    logger.info(f"Starting on 0.0.0.0:{port}")
    logger.info(f"CUDA_VISIBLE_DEVICES={VISIBLE_DEVICES} -> devices={DEVICE_IDS}")
    logger.info(f"Scorer: cuda:{SCORER_DEVICE}, Editors: cuda:{EDITOR_DEVICES}")
    logger.info(f"Leaderboard: {LEADERBOARD_DIR}")

    if args.tunnel:
        from pyngrok import ngrok
        public_url = ngrok.connect(port, "http").public_url
        logger.info(f"Public URL: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=port)

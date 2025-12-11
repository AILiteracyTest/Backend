import os
import time
import uuid
import asyncio
import random
import aiohttp

from io import BytesIO
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not UNSPLASH_ACCESS_KEY:
    raise RuntimeError("UNSPLASH_ACCESS_KEY is not set")

# ========= FastAPI 앱 =========
app = FastAPI(title="Image Analysis (FastAPI+async)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= 전역 세션/클라이언트 =========
http_session: aiohttp.ClientSession | None = None
oai_client: AsyncOpenAI | None = None

@app.on_event("startup")
async def _startup():
    global http_session, oai_client
    timeout = aiohttp.ClientTimeout(total=60)
    http_session = aiohttp.ClientSession(timeout=timeout, trust_env=True)
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@app.on_event("shutdown")
async def _shutdown():
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()

# ========= (캐시/유틸 동일) =========
RUN_TTL_SEC = 15 * 60
run_cache: Dict[str, Dict[str, Any]] = {}

def _gc_runs():
    now = time.time()
    stale = [rid for rid, rec in run_cache.items()
             if now - rec.get("created_at", 0) > RUN_TTL_SEC]
    for rid in stale:
        run_cache.pop(rid, None)

# def build_random_query() -> str:
#     ages = ["teenage", "young", "middle-aged", "elderly"]
#     races = ["white", "black", "asian"]
#     age = random.choice(ages)
#     gender_candidates = ["boy", "girl"] if age in ["teenage", "young"] else ["male", "female"]
#     race = random.choice(races)
#     gender = random.choice(gender_candidates)
#     return f"{age} {race} {gender}"

def build_random_query() -> str:
    """
    사람 / 동물(강아지, 고양이) / 풍경(산, 바다, 사막) 중 하나를 랜덤으로 선택해
    Unsplash 검색용 query 문자열을 만들어 반환한다.
    """
    category = random.choice(["human", "dog", "cat", "landscape"])

    # ---------------- 사람 ----------------
    if category == "human":
        ages = ["teenage", "young", "middle-aged", "elderly"]
        races = ["white", "black", "asian"]
        age = random.choice(ages)
        race = random.choice(races)
        gender_candidates = (
            ["boy", "girl"] if age in ["teenage", "young"] else ["male", "female"]
        )
        gender = random.choice(gender_candidates)
        return f"{age} {race} {gender}"

    # ---------------- 강아지 ----------------
    if category == "dog":
        colors = ["brown dog", "white dog"]
        return random.choice(colors)

    # ---------------- 고양이 ----------------
    if category == "cat":
        colors = ["brown cat", "white cat"]
        return random.choice(colors)

    # ---------------- 풍경 ----------------
    if category == "landscape":
        scenes = ["mountain landscape", "ocean sea view landscape", "desert landscape"]
        return random.choice(scenes)


async def fetch_unsplash_image(query: str) -> List[str]:
    assert http_session is not None
    url = "https://api.unsplash.com/photos/random"
    params = {"client_id": UNSPLASH_ACCESS_KEY, "query": query}
    async with http_session.get(url, params=params) as resp:
        if resp.status != 200:
            txt = await resp.text()
            raise HTTPException(status_code=502, detail=f"Unsplash error: {resp.status}, {txt[:200]}")
        data = await resp.json()
    photos = [data] if isinstance(data, dict) else (data or [])
    return [f"{photos[0]['urls']['raw']}&fm=jpg&w=1080"] if photos else []

async def generate_dalle_image(prompt: str) -> str:
    assert oai_client is not None
    resp = await oai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1792",
        n=1
    )
    return resp.data[0].url

# ========= 응답 모델(간소화) =========
class SyntheticOut(BaseModel):
    generated_image_url: Optional[str] = None

class ImageAnalysisOut(BaseModel):
    mode: str
    run_id: str
    query: str
    unsplash: Dict[str, List[str]]
    synthetic: SyntheticOut

# ========= 라우트 (Sightengine 제거) =========
@app.get("/image_analysis", response_model=ImageAnalysisOut)
async def image_analysis(
    mode: str = Query("default", pattern="^.*$"),  # 하위호환용으로 파라미터만 남김(의미 없음)
    run_id: Optional[str] = Query(None),
):
    """
    Unsplash + DALL·E 3만 생성해서 반환.
    run_id가 있으면 캐시된 같은 이미지/쿼리 재사용, 없으면 새로 생성.
    """
    _gc_runs()

    rec = None
    if run_id:
        rec = run_cache.get(run_id)
        if rec and (time.time() - rec["created_at"] <= RUN_TTL_SEC):
            query = rec["query"]
            real_urls = rec["unsplash"]
            gen_url = rec["gen_url"]
        else:
            rec = None

    if rec is None:
        query = build_random_query()
        if "landscape" in query:
            prompt = f"A high-resolution landscape photo of {query}, natural lighting, clear atmosphere."
        else:
            prompt = (
        f"A realistic portrait of a {query} captured in natural daylight. "
        "Gentle facial expression, smooth lighting, and soft background blur.")
        real_urls, gen_url = await asyncio.gather(
            fetch_unsplash_image(query),
            generate_dalle_image(prompt),
        )
        run_id = uuid.uuid4().hex
        run_cache[run_id] = {
            "created_at": time.time(),
            "query": query,
            "unsplash": real_urls,
            "gen_url": gen_url,
        }

    return ImageAnalysisOut(
        mode=mode,
        run_id=run_id,
        query=query,
        unsplash={"images": real_urls},
        synthetic=SyntheticOut(generated_image_url=gen_url),
    )

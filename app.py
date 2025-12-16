import os
import sys
import time
import uuid
import asyncio
import random
import aiohttp
import sqlite3 #백분위 계산

from io import BytesIO
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pathlib import Path 

from unet_autoencoder.ae_explain import analyze_and_explain

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp_images"
TMP_DIR.mkdir(exist_ok=True)
SCORE_DB_PATH=BASE_DIR/'scores.db' #백분위 계산 - DB 파일 경로 상수 추가

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
    allow_origins=[
        "http://localhost:5173",
        "https://ai-literacy-test.netlify.app"
    ],
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
    await run_in_threadpool(_init_score_db) #백분위 계산-DB 초기화 연결

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
        scenes = ["mountain landscape", "ocean sea view landscape", "desert landscape","forest landscape"]
        return random.choice(scenes)

def _init_score_db() -> None: #백분위 계산-점수 테이블 생성 함수
    conn=sqlite3.connect(SCORE_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()
    finally: 
        conn.close()

def _insert_and_calc(score: int) -> dict: #백분위-DB 저장 + 백분위 계산 함수
    conn = sqlite3.connect(SCORE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # 1) 저장
        conn.execute(
            "INSERT INTO scores (score, created_at) VALUES (?, ?)",
            (score, time.time())
        )
        conn.commit()

        # 2) 전체 점수 기반 통계 (가장 단순)
        rows = conn.execute("SELECT score FROM scores").fetchall()
        scores = [r["score"] for r in rows]
        total = len(scores)

        higher = sum(1 for s in scores if s > score)
        rank = higher + 1
        percentile = round((higher / total) * 100) if total else 0

        return {"rank": rank,"total": total, "percentile": percentile}

    finally:
        conn.close()
        
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

async def download_image(url: str) -> str:
    """
    DALL·E가 생성한 이미지 URL을 받아서 로컬에 저장하고,
    저장된 파일 경로를 문자열로 반환.
    """
    assert http_session is not None

    filename = TMP_DIR / f"{uuid.uuid4().hex}.png"
    async with http_session.get(url) as resp:
        if resp.status != 200:
            txt = await resp.text()
            raise HTTPException(
                status_code=502,
                detail=f"Download error: {resp.status}, {txt[:200]}"
            )
        content = await resp.read()

    with open(filename, "wb") as f:
        f.write(content)

    return str(filename)

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
    explanation: Optional[str] = None

class ImageAnalysisOut(BaseModel):
    mode: str
    run_id: str
    query: str
    unsplash: Dict[str, List[str]]
    synthetic: SyntheticOut
    
class ScoreIn(BaseModel): #백분위 계산-점수 요청 모델
    score: int
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
        # 1) 쿼리 & 프롬프트 생성
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
        
        # 3) DALL·E 이미지 로컬로 다운로드
        local_fake_path = await download_image(gen_url)
        
        # 4) U-Net AE + VLM 설명 실행 
        ae_result = await run_in_threadpool(analyze_and_explain, local_fake_path)
        
        run_id = uuid.uuid4().hex
        run_cache[run_id] = {
            "created_at": time.time(),
            "query": query,
            "unsplash": real_urls,
            "gen_url": gen_url,
            "ae_result": ae_result,
        }
        
    if rec is not None and ae_result is None:
        # 필요 시 재분석 (선택사항)
        local_fake_path = await download_image(gen_url)
        ae_result = await run_in_threadpool(analyze_and_explain, local_fake_path)
        rec["ae_result"] = ae_result
        
    explanation_text = ae_result.get("explanation") if ae_result else None

    return ImageAnalysisOut(
        mode=mode,
        run_id=run_id,
        query=query,
        unsplash={"images": real_urls},
        synthetic=SyntheticOut(generated_image_url=gen_url, explanation=explanation_text,),
    )

@app.post("/score") #백분위 계산-엔트포인트 추가
async def save_score(payload: ScoreIn):
    
    if payload.score < 0:
        raise HTTPException(status_code=400, detail='score must be >= 0')
    
    stats = await run_in_threadpool(_insert_and_calc, payload.score)
    return stats

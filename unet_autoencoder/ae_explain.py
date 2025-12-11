import base64    

from openai import OpenAI

from .ae_core import run_ae


def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_vlm_explanation(overlay_path, mean_err, p95_err, max_err):
    img_b64 = load_image_base64(overlay_path)

    system_prompt = """
    당신의 역할은 '이미지가 왜 AI가 생성한 이미지처럼 보이는지'
    Autoencoder 재구성 오차(heatmap)를 근거로 명확하게 설명하는 것입니다.

    제공되는 합성 이미지에는 다음 네 가지가 포함됩니다:
    1) 원본 이미지
    2) Autoencoder 재구성 이미지
    3) 재구성 오차 히트맵 (파랑=오차 낮음, 빨강=오차 큼)
    4) 히트맵을 원본 위에 덮은 오버레이 이미지

    Autoencoder는 '실제 사진(real-world photos)'만을 학습한 모델이며,
    특정 영역에서 재구성 오차가 크다는 것은
    그 구역이 실제 사진 분포에서 벗어난 비정상적 패턴을 포함한다는 것을 의미합니다.

    당신의 답변에서는 다음을 지켜주세요:
    - 구체적인 위치를 언급하세요.
    - '왜 해당 영역이 비정상적인지'를 사진적 관점(텍스처, 형태, 조명, 구조 등)에서 설명하세요.
    - 출력 언어는 반드시 한국어로 하세요.
    """

    user_text = f"""
    아래 이미지는 원본/재구성/오차 히트맵/오버레이를 하나로 합친 이미지입니다.

    Autoencoder 재구성 오차 통계값은 다음과 같습니다:
    - 평균 오차(mean error): {mean_err:.6f}
    - 95퍼센타일 오차(p95 error): {p95_err:.6f}
    - 최대 오차(max error): {max_err:.6f}

    다음을 설명해주세요:
    1) 이미지에서 재구성 오차가 큰(빨간색) 영역이 구체적으로 어디인지.
    2) 그 영역들이 왜 AI가 생성한 이미지인지.
    """

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",   
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    },
                ],
            },
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content

def analyze_and_explain(img_path: str, ckpt_name: str = "model20.pth"):
    """
    외부(예: app.py)에서 한 번에 쓰기 좋은 통합 함수.
    1) AE로 heatmap/통계 계산
    2) VLM으로 '왜 가짜처럼 보이는지' 설명
    """
    ae_result = run_ae(img_path, ckpt_name=ckpt_name)

    explanation = ask_vlm_explanation(
        ae_result["overlay_path"],
        ae_result["mean_err"],
        ae_result["p95_err"],
        ae_result["max_err"],
    )

    ae_result["explanation"] = explanation
    return ae_result
# 强基通后端 - Vercel原生ASGI版（无mangum）

import os
import re
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI(title="强基通 API", version="0.4.0")

# ============ 配置 ============
KB_DIR = Path("/var/task/knowledge_base")
if not KB_DIR.exists():
    KB_DIR = Path(__file__).parent.parent / "knowledge_base"

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-lite-4k")
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# ============ 数据模型 ============
class MatchRequest(BaseModel):
    province: str
    subjects: List[str]
    score: int

class QueryRequest(BaseModel):
    school_name: str

class ChatRequest(BaseModel):
    message: str

# ============ 知识库 ============
SCHOOLS_CACHE = None

def load_all_schools():
    global SCHOOLS_CACHE
    if SCHOOLS_CACHE is not None:
        return SCHOOLS_CACHE
    schools = []
    if not KB_DIR.exists():
        return schools
    for md_file in KB_DIR.glob("*.md"):
        if md_file.name.startswith("00_") or md_file.name.startswith("03_"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            schools.append(parse_school_md(md_file.stem, content))
        except Exception as e:
            print(f"[ERROR] {md_file.name}: {e}")
    SCHOOLS_CACHE = schools
    return schools

def parse_school_md(name: str, content: str):
    info = {"name": name, "raw": content}
    subj_match = re.search(r'## 选科要求\s*\n(.*?)(?=\n## |$)', content, re.S)
    info["subject_requirement"] = subj_match.group(1).strip() if subj_match else ""
    major_match = re.search(r'## 招生专业\s*\n(.*?)(?=\n## |$)', content, re.S)
    info["majors"] = major_match.group(1).strip() if major_match else ""
    exam_match = re.search(r'## 校考形式\s*\n(.*?)(?=\n## |$)', content, re.S)
    info["exam_form"] = exam_match.group(1).strip() if exam_match else ""
    note_match = re.search(r'## 备注与特殊政策\s*\n(.*?)(?=\n## |$)', content, re.S)
    info["notes"] = note_match.group(1).strip() if note_match else ""
    return info

def get_school_by_name(name: str):
    if KB_DIR is None:
        return None
    for md_file in KB_DIR.glob("*.md"):
        if md_file.stem == name:
            content = md_file.read_text(encoding="utf-8")
            return parse_school_md(name, content)
    return None

# ============ 分数线数据 ============
CUT_OFF_2024 = {
    "四川": {
        "清华大学": {"基础理科类": 678},
        "北京大学": {"数学类": 677},
        "国防科技大学": {"数学/物理": 635},
        "重庆大学": {"数学组": 631, "物理组": 641},
        "西北工业大学": {"航空航天类": 650},
        "中国农业大学": {"生物科学": 624},
        "中国海洋大学": {"海洋科学": 655.6, "生物科学": 612, "海洋技术": 640.4},
        "四川大学": {"数学类": 645, "物理学类": 640},
        "西安交通大学": {"数学类": 650},
        "哈尔滨工业大学": {"数学类": 660},
        "华中科技大学": {"数学类": 640},
        "南开大学": {"数学类": 650},
        "武汉大学": {"数学类": 650},
        "中南大学": {"数学类": 630},
        "电子科技大学": {"信息与计算科学": 640},
    }
}

FU_JIAO_NAN = ["复旦大学", "上海交通大学", "南京大学", "浙江大学",
                "中国科学技术大学", "西安交通大学", "同济大学", "厦门大学",
                "兰州大学", "北京航空航天大学"]

def estimate_cutoff(province: str, school_name: str):
    province_data = CUT_OFF_2024.get(province, {})
    school_data = province_data.get(school_name)
    if school_data:
        return min(school_data.values())
    return None

def check_subject_match(school_info: dict, subjects: List[str]):
    req = school_info.get("subject_requirement", "").lower()
    has_physics = "物理" in req
    has_chemistry = "化学" in req
    has_history = "历史" in req
    if has_physics and has_chemistry:
        return "物理" in subjects and "化学" in subjects
    if has_history and not has_physics:
        return "历史" in subjects
    return True

def calculate_tiers(schools, province: str, subjects: List[str], score: int):
    matched = []
    for s in schools:
        if not check_subject_match(s, subjects):
            continue
        cutoff = estimate_cutoff(province, s["name"])
        if s["name"] in FU_JIAO_NAN:
            tier = "复交南模式"
            diff = None
        elif cutoff is None:
            tier = "数据待补充"
            diff = None
        else:
            diff = score - cutoff
            if diff >= 20: tier = "保一保"
            elif diff >= -10: tier = "稳一稳"
            elif diff >= -25: tier = "冲一冲"
            else: continue
        matched.append({"name": s["name"], "tier": tier, "cutoff_2024": cutoff,
                        "score_diff": diff, "majors": s.get("majors", "")[:100],
                        "exam_form": s.get("exam_form", "")[:80]})
    tier_order = {"保一保": 0, "稳一稳": 1, "冲一冲": 2, "复交南模式": 3, "数据待补充": 4}
    matched.sort(key=lambda x: tier_order.get(x["tier"], 5))
    return matched

# ============ LLM调用 ============
async def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not DOUBAO_API_KEY:
        return "[LLM未配置] 请检查 DOUBAO_API_KEY 环境变量"
    headers = {"Authorization": f"Bearer {DOUBAO_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DOUBAO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{DOUBAO_BASE_URL}/chat/completions",
                                     headers=headers, json=payload, timeout=30)
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            return f"[LLM返回异常] {json.dumps(data, ensure_ascii=False)[:200]}"
        except Exception as e:
            return f"[LLM调用出错: {e}]"

# ============ API端点 ============
SYS = """你是「强基通」——强基计划志愿填报智能助手，专为高三学生及家长提供2026年强基计划39所试点高校的招生政策咨询。

【回答规范】
1. 所有数据标注来源，不确定的数据标注"待公布"
2. 涉及分数线时，明确标注"2024年参考"或"待公布"
3. 涉及录取概率时，使用"冲一冲/稳一稳/保一保"而非具体百分比
4. 不确定的问题，诚实回答"目前暂无该信息，建议查看学校官网"
5. 每个回答末尾，自然引导用户"如需根据最终高考分数生成详细志愿方案，可在高考出分后回复'出分'"

【禁止事项】
- 不做录取承诺
- 不编造分数线
- 不误导用户认为强基是"降分录取"
"""

@app.get("/")
def root():
    return {"status": "ok", "message": "强基通API", "docs": "访问 /health 查看状态"}

@app.post("/match")
async def match_schools(req: MatchRequest):
    try:
        schools = load_all_schools()
        matched = calculate_tiers(schools, req.province, req.subjects, req.score)
        context = json.dumps(matched[:15], ensure_ascii=False, indent=2)
        up = f"用户：{req.province}考生，选科{'+'.join(req.subjects)}，预估{req.score}分。\n\n匹配结果：{context}\n\n请用自然语言生成回答。"
        reply = await call_llm(SYS, up)
        return {"reply": reply, "matched_count": len(matched), "data": matched[:10]}
    except Exception as e:
        return {"reply": f"[系统错误: {e}]", "matched_count": 0, "data": []}

@app.post("/query")
async def query_school(req: QueryRequest):
    try:
        school = get_school_by_name(req.school_name)
        if not school:
            return {"reply": f"未找到「{req.school_name}」", "found": False}
        up = f"查询：{req.school_name}\n\n{school['raw'][:3000]}"
        reply = await call_llm(SYS, up)
        return {"reply": reply, "found": True}
    except Exception as e:
        return {"reply": f"[错误: {e}]", "found": False}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        reply = await call_llm(SYS, req.message)
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"[错误: {e}]"}

@app.get("/health")
def health_check():
    sc = load_all_schools()
    return {"status": "ok", "kb_loaded": len(sc), "kb_dir": str(KB_DIR) if KB_DIR else "null"}

# Vercel原生ASGI检测：不需要mangum，直接暴露app变量即可

# 强基通后端 - Vercel Serverless Functions 兼容版
# FastAPI + 本地知识库检索 + 豆包LLM + mangum适配器

import os
import re
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

# ============ mangum: FastAPI ↔ Vercel 桥梁 ============
# Vercel 运行的是 AWS Lambda 风格的函数，不是普通服务器
# mangum 把 Lambda 事件转换成 ASGI 请求，让 FastAPI 能正常工作
from mangum import Mangum

app = FastAPI(title="强基通 API", version="0.4.0")

# ============ 配置 ============
# Vercel 部署时文件位置：
#   api/index.py  →  /var/task/api/index.py
#   knowledge_base/ → /var/task/knowledge_base/

KB_DIR_CANDIDATES = [
    Path("/var/task/knowledge_base"),                    # Vercel生产环境
    Path(__file__).parent.parent / "knowledge_base",     # 本地: api/../knowledge_base
    Path(__file__).parent / "knowledge_base",            # 本地备选
    Path.cwd() / "knowledge_base",                       # cwd
]

KB_DIR = None
for candidate in KB_DIR_CANDIDATES:
    if candidate.exists():
        KB_DIR = candidate
        break

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-lite-4k")
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

@app.get("/")
def root():
    return {"status": "ok", "message": "强基通API", "docs": "访问 /health 查看状态"}

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
    if KB_DIR is None or not KB_DIR.exists():
        print(f"[WARN] KB_DIR not found: {KB_DIR}")
        return schools
    for md_file in sorted(KB_DIR.glob("*.md")):
        name = md_file.stem
        if name.startswith("00_") or name.startswith("03_"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            schools.append(parse_school_md(name, content))
        except Exception as e:
            print(f"[ERROR] {md_file.name}: {e}")
    SCHOOLS_CACHE = schools
    return schools

def parse_school_md(name: str, content: str):
    """解析单校Markdown（新版8章节格式），提取结构化信息"""
    info = {"name": name, "raw": content}
    
    # 提取选科要求（**选科要求**：...）
    subj_match = re.search(r'\*\*选科要求\*\*：\s*(.*?)(?=\n\*\*|$)', content, re.S)
    info["subject_requirement"] = subj_match.group(1).strip() if subj_match else ""
    
    # 提取招生专业（"## 2026年招生专业"章节的第一段）
    major_match = re.search(r'## 2026年招生专业\s*\n(.*?)(?=\n\*\*选科要求|\n## |$)', content, re.S)
    info["majors"] = major_match.group(1).strip() if major_match else ""
    
    # 提取综评公式（"## 综合成绩折算"章节）
    formula_match = re.search(r'## 综合成绩折算\s*\n(.*?)(?=\n## |$)', content, re.S)
    info["formula"] = formula_match.group(1).strip() if formula_match else ""
    
    # 提取校考形式
    exam_match = re.search(r'## 校考形式\s*\n(.*?)(?=\n## |$)', content, re.S)
    info["exam_form"] = exam_match.group(1).strip() if exam_match else ""
    
    # 提取备注（"**破格条件**："）
    note_match = re.search(r'\*\*破格条件\*\*：\s*(.*?)(?=\n## |$)', content, re.S)
    info["notes"] = note_match.group(1).strip() if note_match else ""
    
    # 提取往年分数线
    cutoff_match = re.search(r'## 往年入围分数线参考\s*\n(.*?)(?=\n## |$)', content, re.S)
    info["cutoff_history"] = cutoff_match.group(1).strip() if cutoff_match else ""
    
    return info

def get_school_by_name(name: str):
    for s in load_all_schools():
        if s["name"] == name:
            return s
    return None

# ============ 分数线数据（2025年参考，网络收集版） ============
CUT_OFF_2024 = {
    "四川": {
        "清华大学": {"参考入围线": 678.0}, "北京大学": {"参考入围线": 677.0},
        "国防科技大学": {"参考入围线": 635.0},
        "重庆大学": {"数学组": 631.0, "物理组": 641.0},
        "西北工业大学": {"参考入围线": 650.0},
        "中国农业大学": {"参考入围线": 624.0},
        "中国海洋大学": {"海洋科学": 655.6, "生物科学": 612.0, "海洋技术": 640.4},
    },
    "北京": {
        "清华大学": {"参考入围线": 670.0},
        "北京大学": {"Ⅰ组": 670.0, "Ⅱ组": 679.0},
        "西北工业大学": {"参考入围线": 636.0},
        "中国农业大学": {"参考入围线": 627.0},
    },
    "上海": {
        "清华大学": {"参考入围线": 604.0}, "北京大学": {"参考入围线": 606.0},
    },
    "天津": {
        "清华大学": {"参考入围线": 688.0}, "北京大学": {"参考入围线": 691.0},
        "华南理工大学": {"数学类": 685.9}, "重庆大学": {"数学组": 620.0},
        "中国农业大学": {"参考入围线": 639.0},
    },
    "重庆": {
        "北京大学": {"参考入围线": 677.0},
        "重庆大学": {"数学组": 613.0, "物理组": 628.0},
        "中国农业大学": {"参考入围线": 605.0},
    },
    "广东": {
        "清华大学": {"参考入围线": 683.0}, "北京大学": {"参考入围线": 681.0},
        "华南理工大学": {"数学类": 665.7, "生物技术": 613.0},
        "重庆大学": {"数学组": 619.0, "物理组": 622.0},
        "中国农业大学": {"参考入围线": 612.0},
    },
    "江苏": {
        "清华大学": {"参考入围线": 670.0},
        "北京大学": {"Ⅰ组": 672.0, "Ⅱ组": 666.0},
        "华中科技大学": {"物理学": 645.0},
        "重庆大学": {"数学组": 638.0, "物理组": 635.0},
        "西北工业大学": {"参考入围线": 649.0},
        "中国农业大学": {"参考入围线": 627.0},
    },
    "山东": {
        "清华大学": {"参考入围线": 679.0},
        "北京大学": {"Ⅰ组": 677.0, "Ⅱ组": 671.0},
        "国防科技大学": {"参考入围线": 628.0},
        "重庆大学": {"数学组": 609.0, "物理组": 625.0},
        "西北工业大学": {"参考入围线": 641.0},
        "中国农业大学": {"参考入围线": 628.0},
    },
    "浙江": {
        "北京大学": {"Ⅰ组": 684.0, "Ⅱ组": 690.0},
        "重庆大学": {"数学组": 641.0, "物理组": 647.0},
        "西北工业大学": {"参考入围线": 654.0},
        "中国农业大学": {"参考入围线": 637.0},
    },
    "安徽": {
        "清华大学": {"参考入围线": 681.0}, "北京大学": {"参考入围线": 682.0},
        "重庆大学": {"数学组": 630.0, "物理组": 642.0},
        "西北工业大学": {"参考入围线": 650.0},
        "中国农业大学": {"参考入围线": 621.0},
    },
    "河北": {
        "清华大学": {"参考入围线": 680.0}, "北京大学": {"参考入围线": 677.0},
        "华中科技大学": {"基础医学": 628.0},
        "重庆大学": {"数学组": 626.0, "物理组": 635.0},
        "西北工业大学": {"参考入围线": 651.0},
        "中国农业大学": {"参考入围线": 627.0},
    },
    "河南": {
        "国防科技大学": {"参考入围线": 635.0},
        "重庆大学": {"数学组": 645.0, "物理组": 651.0},
        "西北工业大学": {"参考入围线": 657.0},
        "中国农业大学": {"参考入围线": 632.0},
    },
    "湖北": {
        "重庆大学": {"数学组": 620.0},
        "西北工业大学": {"参考入围线": 625.0},
        "中国农业大学": {"参考入围线": 587.0},
    },
    "湖南": {
        "国防科技大学": {"参考入围线": 625.0},
        "重庆大学": {"数学组": 618.0, "物理组": 620.0},
        "西北工业大学": {"参考入围线": 635.0},
        "中国农业大学": {"参考入围线": 598.0},
    },
    "辽宁": {
        "清华大学": {"参考入围线": 677.0}, "北京大学": {"参考入围线": 677.0},
        "国防科技大学": {"参考入围线": 629.0},
        "中国农业大学": {"参考入围线": 625.0},
    },
    "吉林": {
        "清华大学": {"参考入围线": 671.0}, "北京大学": {"参考入围线": 673.0},
        "吉林大学": {"数学": 609.0, "物理": 616.0, "化学": 610.0, "古文字学": 613.0},
        "中国农业大学": {"参考入围线": 610.0},
    },
    "黑龙江": {
        "清华大学": {"参考入围线": 668.0}, "北京大学": {"参考入围线": 672.0},
        "中国农业大学": {"参考入围线": 607.0},
    },
    "福建": {
        "清华大学": {"理科": 676.0, "文科": 645.0},
        "重庆大学": {"数学组": 620.0},
        "中国农业大学": {"参考入围线": 611.0},
    },
    "江西": {
        "国防科技大学": {"参考入围线": 611.0},
        "重庆大学": {"数学组": 601.0},
        "中国农业大学": {"参考入围线": 582.0},
    },
    "山西": {
        "华南理工大学": {"数学类": 648.5, "化学类": 623.0, "生物技术": 624.0},
        "重庆大学": {"数学组": 616.0, "物理组": 634.0},
        "中国农业大学": {"参考入围线": 614.0},
    },
    "陕西": {
        "国防科技大学": {"参考入围线": 608.0},
        "重庆大学": {"数学组": 620.0, "物理组": 633.0},
        "西北工业大学": {"参考入围线": 652.0},
        "中国农业大学": {"参考入围线": 603.0},
    },
    "云南": {
        "重庆大学": {"数学组": 629.0},
        "中国农业大学": {"参考入围线": 606.0},
    },
    "贵州": {
        "国防科技大学": {"参考入围线": 595.0},
        "重庆大学": {"数学组": 606.0, "物理组": 612.0},
        "中国农业大学": {"参考入围线": 601.0},
    },
    "甘肃": {
        "重庆大学": {"数学组": 591.0},
        "中国农业大学": {"参考入围线": 593.0},
    },
    "广西": {
        "重庆大学": {"数学组": 593.0},
        "中国农业大学": {"参考入围线": 582.0},
    },
    "内蒙古": {
        "清华大学": {"参考入围线": 683.0}, "北京大学": {"参考入围线": 678.0},
        "中国农业大学": {"参考入围线": 627.0},
    },
    "海南": {
        "北京大学": {"参考入围线": 807.0},
    },
    "新疆": {
        "中国农业大学": {"参考入围线": 555.0},
    },
}

FU_JIAO_NAN = ["复旦大学", "上海交通大学", "南京大学", "浙江大学",
                "中国科学技术大学", "西安交通大学", "同济大学", "厦门大学",
                "兰州大学", "北京航空航天大学"]

def estimate_cutoff(province: str, school_name: str):
    pd = CUT_OFF_2024.get(province, {})
    sd = pd.get(school_name)
    if sd: return min(sd.values())
    return None

def check_subject_match(school_info: dict, subjects: List[str]):
    req = school_info.get("subject_requirement", "").lower()
    hp, hc, hh = "物理" in req, "化学" in req, "历史" in req
    if hp and hc: return "物理" in subjects and "化学" in subjects
    if hh and not hp: return "历史" in subjects
    return True

def calculate_tiers(schools, province: str, subjects: List[str], score: int):
    matched = []
    for s in schools:
        if not check_subject_match(s, subjects): continue
        cutoff = estimate_cutoff(province, s["name"])
        if s["name"] in FU_JIAO_NAN:
            tier, diff = "复交南模式（高考出分前校考）", None
        elif cutoff is None:
            tier, diff = "数据待补充", None
        else:
            diff = score - cutoff
            if diff >= 20: tier = "保一保"
            elif diff >= -10: tier = "稳一稳"
            elif diff >= -25: tier = "冲一冲"
            else: continue
        matched.append({"name": s["name"], "tier": tier, "cutoff_2024": cutoff,
                        "score_diff": diff, "majors": s.get("majors", "")[:100],
                        "exam_form": s.get("exam_form", "")[:80]})
    tier_order = {"保一保": 0, "稳一稳": 1, "冲一冲": 2,
                  "复交南模式（高考出分前校考）": 3, "数据待补充": 4}
    matched.sort(key=lambda x: tier_order.get(x["tier"], 5))
    return matched

# ============ LLM ============
async def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not DOUBAO_API_KEY:
        return "[LLM未配置] 请检查 DOUBAO_API_KEY"
    headers = {"Authorization": f"Bearer {DOUBAO_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": DOUBAO_MODEL, "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], "temperature": 0.7, "max_tokens": 2048}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{DOUBAO_BASE_URL}/chat/completions",
                                     headers=headers, json=payload, timeout=30)
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            return f"[LLM异常] {json.dumps(data, ensure_ascii=False)[:200]}"
        except Exception as e:
            return f"[LLM出错: {e}]"

# ============ API端点 ============
SYS = "你是「强基通」强基计划志愿填报助手。数据标注来源，不确定标注'待公布'，分数线标注'2024参考'，末尾引导'高考出分后回复出分获取精准匹配'。"

@app.post("/match")
async def match_schools(req: MatchRequest):
    try:
        schools = load_all_schools()
        matched = calculate_tiers(schools, req.province, req.subjects, req.score)
        ctx = json.dumps(matched[:15], ensure_ascii=False, indent=2)
        up = f"用户：{req.province}考生，选科{'+'.join(req.subjects)}，预估{req.score}分。\n\n匹配结果：{ctx}\n\n请生成回答，含：1.冲稳保分层 2.学校核心信息 3.注意事项 4.末尾引导。"
        reply = await call_llm(SYS, up)
        return {"reply": reply, "matched_count": len(matched), "data": matched[:10]}
    except Exception as e:
        return {"reply": f"[错误: {e}]", "matched_count": 0, "data": []}

@app.post("/query")
async def query_school(req: QueryRequest):
    try:
        school = get_school_by_name(req.school_name)
        if not school:
            return {"reply": f"未找到「{req.school_name}」", "found": False}
        up = f"查询：{req.school_name}\n\n{school['raw'][:3000]}\n\n请生成完整介绍。"
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
    return {"status": "ok", "kb_loaded": len(sc)}

# ============ Vercel 入口 ============
# mangum 把 FastAPI app 包装成 Vercel 能识别的 Lambda handler
handler = Mangum(app, lifespan="off")

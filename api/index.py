# 强基通后端 API
# FastAPI + 本地知识库检索 + 豆包LLM

import os
import re
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

# ============ mangum: Vercel Lambda adapter ============
from mangum import Mangum

app = FastAPI(title="强基通 API", version="0.1.0")

# ============ 配置 ============
# 自动检测知识库路径（本地开发 vs Vercel部署）
# Vercel Serverless: api/index.py -> 知识库在 ../knowledge_base
KB_DIR_CANDIDATES = [
    Path(__file__).parent / "knowledge_base",           # api/knowledge_base
    Path(__file__).parent.parent / "knowledge_base",    # 根目录/knowledge_base
    Path(__file__).parent / "knowledge_base_v2",        # api/knowledge_base_v2
    Path(__file__).parent.parent / "knowledge_base_v2", # 根目录/knowledge_base_v2
    Path.cwd() / "knowledge_base",                      # cwd/knowledge_base
    Path.cwd() / "knowledge_base_v2",                   # cwd/knowledge_base_v2
]

KB_DIR = None
for candidate in KB_DIR_CANDIDATES:
    if candidate.exists():
        KB_DIR = candidate
        break

if KB_DIR is None:
    raise RuntimeError("知识库目录 knowledge_base_v2 未找到")

# ============ 数据模型 ============
class MatchRequest(BaseModel):
    province: str          # 省份，如"四川"
    subjects: List[str]    # 选科，如["物理", "化学"]
    score: int             # 预估分数

class QueryRequest(BaseModel):
    school_name: str       # 学校名，如"四川大学"

class ChatRequest(BaseModel):
    message: str           # 用户原始消息
    history: Optional[List[dict]] = []  # 对话历史

# ============ 知识库读取 ============

def load_all_schools():
    """加载所有学校知识库"""
    schools = []
    if not KB_DIR.exists():
        return schools
    for md_file in KB_DIR.glob("*.md"):
        if md_file.name.startswith("00_") or md_file.name.startswith("03_"):
            continue
        content = md_file.read_text(encoding="utf-8")
        schools.append(parse_school_md(md_file.stem, content))
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
    """按名称查找学校"""
    for md_file in KB_DIR.glob("*.md"):
        if md_file.stem == name:
            content = md_file.read_text(encoding="utf-8")
            return parse_school_md(name, content)
    return None

# ============ 分数线数据（2025年参考，网络收集版） ============
# 数据来源：高考100、北京高考在线、自主选拔在线等网络汇总
# 标注：①清华北大等采用高考裸分入围；②部分学校采用加权成绩；③复交南模式无入围线
# 提示：其他省份/学校数据待补充，匹配时无分数线学校标注"数据待补充"
CUT_OFF_2024 = {
    "四川": {
        "清华大学": {"参考入围线": 678.0},
        "北京大学": {"参考入围线": 677.0},
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
        "清华大学": {"参考入围线": 604.0},
        "北京大学": {"参考入围线": 606.0},
    },
    "天津": {
        "清华大学": {"参考入围线": 688.0},
        "北京大学": {"参考入围线": 691.0},
        "华南理工大学": {"数学类": 685.9},
        "重庆大学": {"数学组": 620.0},
        "中国农业大学": {"参考入围线": 639.0},
    },
    "重庆": {
        "北京大学": {"参考入围线": 677.0},
        "重庆大学": {"数学组": 613.0, "物理组": 628.0},
        "中国农业大学": {"参考入围线": 605.0},
    },
    "广东": {
        "清华大学": {"参考入围线": 683.0},
        "北京大学": {"参考入围线": 681.0},
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
        "清华大学": {"参考入围线": 681.0},
        "北京大学": {"参考入围线": 682.0},
        "重庆大学": {"数学组": 630.0, "物理组": 642.0},
        "西北工业大学": {"参考入围线": 650.0},
        "中国农业大学": {"参考入围线": 621.0},
    },
    "河北": {
        "清华大学": {"参考入围线": 680.0},
        "北京大学": {"参考入围线": 677.0},
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
        "清华大学": {"参考入围线": 677.0},
        "北京大学": {"参考入围线": 677.0},
        "国防科技大学": {"参考入围线": 629.0},
        "中国农业大学": {"参考入围线": 625.0},
    },
    "吉林": {
        "清华大学": {"参考入围线": 671.0},
        "北京大学": {"参考入围线": 673.0},
        "吉林大学": {"数学": 609.0, "物理": 616.0, "化学": 610.0, "古文字学": 613.0},
        "中国农业大学": {"参考入围线": 610.0},
    },
    "黑龙江": {
        "清华大学": {"参考入围线": 668.0},
        "北京大学": {"参考入围线": 672.0},
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
        "清华大学": {"参考入围线": 683.0},
        "北京大学": {"参考入围线": 678.0},
        "中国农业大学": {"参考入围线": 627.0},
    },
    "海南": {
        "北京大学": {"参考入围线": 807.0},
    },
    "新疆": {
        "中国农业大学": {"参考入围线": 555.0},
    },
}

# 复交南模式学校（无入围线，高考出分前校考）
FU_JIAO_NAN = ["复旦大学", "上海交通大学", "南京大学", "浙江大学", 
                "中国科学技术大学", "西安交通大学", "同济大学", "厦门大学",
                "兰州大学", "北京航空航天大学"]

def estimate_cutoff(province: str, school_name: str):
    """估算入围线：优先用2024数据，无则返回None"""
    province_data = CUT_OFF_2024.get(province, {})
    school_data = province_data.get(school_name)
    if school_data:
        # 取所有专业中的最低入围线
        return min(school_data.values())
    return None

def check_subject_match(school_info: dict, subjects: List[str]):
    """检查选科是否匹配"""
    req = school_info.get("subject_requirement", "")
    req_lower = req.lower()
    
    # 简单匹配逻辑
    has_physics = "物理" in req_lower
    has_chemistry = "化学" in req_lower
    has_history = "历史" in req_lower
    
    user_has_physics = "物理" in subjects
    user_has_chemistry = "化学" in subjects
    user_has_history = "历史" in subjects
    
    # 理工类需要物理+化学
    if has_physics and has_chemistry:
        return user_has_physics and user_has_chemistry
    # 文史类需要历史
    if has_history and not has_physics:
        return user_has_history
    # 如果选科要求不明确，默认匹配
    return True

# ============ 匹配算法 ============

def calculate_tiers(schools, province: str, subjects: List[str], score: int):
    """冲/稳/保分层"""
    matched = []
    
    for s in schools:
        # 选科匹配
        if not check_subject_match(s, subjects):
            continue
        
        # 获取入围线
        cutoff = estimate_cutoff(province, s["name"])
        
        # 判断层次
        if s["name"] in FU_JIAO_NAN:
            tier = "复交南模式（高考出分前校考）"
            diff = None
        elif cutoff is None:
            tier = "数据待补充"
            diff = None
        else:
            diff = score - cutoff
            if diff >= 20:
                tier = "保一保"
            elif diff >= -10:
                tier = "稳一稳"
            elif diff >= -25:
                tier = "冲一冲"
            else:
                continue  # 分数差距太大，不推荐
        
        matched.append({
            "name": s["name"],
            "tier": tier,
            "cutoff_2024": cutoff,
            "score_diff": diff,
            "majors": s.get("majors", "")[:100] + "..." if len(s.get("majors", "")) > 100 else s.get("majors", ""),
            "exam_form": s.get("exam_form", "")[:80] + "..." if len(s.get("exam_form", "")) > 80 else s.get("exam_form", ""),
        })
    
    # 按层次排序
    tier_order = {"保一保": 0, "稳一稳": 1, "冲一冲": 2, "复交南模式（高考出分前校考）": 3, "数据待补充": 4}
    matched.sort(key=lambda x: tier_order.get(x["tier"], 5))
    return matched

# ============ LLM调用（豆包） ============

async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用豆包LLM生成回答"""
    if not DOUBAO_API_KEY:
        return "[LLM未配置] " + user_prompt[:200]
    
    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json"
    }
    
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
            resp = await client.post(
                f"{DOUBAO_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM调用出错: {e}] 请检查API Key配置"

# ============ API端点 ============

@app.post("/match")
async def match_schools(req: MatchRequest):
    """智能匹配：根据省份、选科、分数推荐学校"""
    schools = load_all_schools()
    matched = calculate_tiers(schools, req.province, req.subjects, req.score)
    
    # 构建Prompt让LLM生成自然语言回答
    context = json.dumps(matched[:15], ensure_ascii=False, indent=2)
    
    system_prompt = """你是「强基通」志愿填报助手。根据提供的学校匹配数据，生成一段自然、有用的回答。
规则：
1. 数据标注"2024年参考"或"待公布"
2. 不使用具体录取概率百分比
3. 最后自然引导"高考出分后回复'出分'获取精准方案"
4. 语气亲切，适合高三家长阅读"""
    
    user_prompt = f"""用户情况：{req.province}考生，选科{'+'.join(req.subjects)}，预估{req.score}分。
匹配结果（JSON）：
{context}

请生成一段自然的回答，包含冲/稳/保三层推荐，每层列2-3所学校，说明入围线和校考形式。"""
    
    reply = await call_llm(system_prompt, user_prompt)
    
    return {
        "reply": reply,
        "matched_count": len(matched),
        "data": matched[:10]
    }

@app.post("/query")
async def query_school(req: QueryRequest):
    """单校查询：返回学校完整详情"""
    school = get_school_by_name(req.school_name)
    if not school:
        return {"reply": "知识库中未找到「" + req.school_name + "」的信息。请确认学校名称是否正确（如四川大学）。", "found": False}
    
    system_prompt = "你是「强基通」志愿填报助手。根据提供的学校信息，生成结构化、易读的回答。"
    user_prompt = f"用户查询：{req.school_name}的强基计划详情。\n\n学校数据：\n{school['raw'][:3000]}\n\n请生成一段结构清晰的回答，包含招生专业、入围规则、校考形式、录取规则。"
    
    reply = await call_llm(system_prompt, user_prompt)
    
    return {
        "reply": reply,
        "found": True,
        "school_name": school["name"]
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    """通用对话：直接透传给LLM，附加知识库上下文"""
    # 简单意图识别
    msg = req.message
    
    # 如果是分数匹配格式
    score_match = re.search(r'(\d{3})\s*分', msg)
    province_match = re.search(r'(北京|上海|天津|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)', msg)
    
    if score_match and province_match:
        # 提取分数和省份，调用匹配
        score = int(score_match.group(1))
        province = province_match.group(1)
        # 简单选科判断
        subjects = []
        if "物理" in msg:
            subjects.append("物理")
        if "化学" in msg:
            subjects.append("化学")
        if "历史" in msg:
            subjects.append("历史")
        if "生物" in msg:
            subjects.append("生物")
        if "政治" in msg:
            subjects.append("政治")
        if "地理" in msg:
            subjects.append("地理")
        if not subjects:
            subjects = ["物理", "化学"]  # 默认
        
        return await match_schools(MatchRequest(province=province, subjects=subjects, score=score))
    
    # 通用对话
    system_prompt = """你是「强基通」——强基计划志愿填报智能助手。
规则：
1. 必须基于知识库信息回答，不确定则说"暂无该信息"
2. 涉及分数线标注"2024年参考"
3. 不做录取承诺
4. 最后引导"高考出分后回复'出分'" """
    
    reply = await call_llm(system_prompt, msg)
    return {"reply": reply}

@app.get("/health")
def health_check():
    return {"status": "ok", "kb_loaded": len(load_all_schools())}

# ============ Vercel entry point ============
handler = Mangum(app, lifespan="off")

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
import re
import json
import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import openai
from api.utils.logging import logger

QuestionType = Literal["mcq", "saq", "caselet", "coding"]
DifficultyLevel = Literal["easy", "medium", "hard"]

DEFAULT_JD_SKILL_MAP: Dict[str, List[str]] = {
    "sql": ["sql", "query", "joins", "etl", "window function", "database"],
    "metrics": ["kpi", "metric", "retention", "conversion", "funnel", "a/b"],
    "product_thinking": ["product", "roadmap", "tradeoff", "prioritization", "user journey"],
    "python": ["python", "pandas", "numpy"],
    "statistics": ["statistics", "hypothesis", "significance", "regression"],
    "communication": ["communication", "stakeholder", "presentation", "storytelling"],
}

DEFAULT_DIFFICULTY_BY_LEVEL: Dict[str, Dict[DifficultyLevel, float]] = {
    "beginner": {"easy": 0.50, "medium": 0.40, "hard": 0.10},
    "intermediate": {"easy": 0.25, "medium": 0.55, "hard": 0.20},
    "advanced": {"easy": 0.15, "medium": 0.45, "hard": 0.40},
}

DEFAULT_TYPE_SPLIT_BY_MODE: Dict[str, Dict[QuestionType, int]] = {
    "curriculum": {"mcq": 10, "saq": 4, "coding": 2, "caselet": 2},
    "jd": {"mcq": 15, "saq": 5, "caselet": 1},
}


@dataclass
class Blueprint:
    mode: Literal["curriculum", "jd"]
    difficulty_distribution: Dict[DifficultyLevel, float]
    type_distribution: Dict[QuestionType, int]
    skill_targets: Dict[str, float]


def _normalize_skill_name(skill: str) -> str:
    return re.sub(r"\s+", "_", skill.strip().lower())


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


async def _generate_question_with_llm(
    mode: str,
    qtype: QuestionType,
    difficulty: DifficultyLevel,
    skill: str,
    context_title: str,
    module_name: Optional[str],
    idx: int,
) -> Dict[str, Any]:
    """Generate a single question using OpenAI LLM."""

    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key) if api_key else None

    if not client:
        logger.warning("OPENAI_API_KEY not configured; using fallback question generation")
        if qtype == "mcq":
            return {
                "stem": f"[{context_title}] Which option best demonstrates {skill.replace('_', ' ')}? (Q{idx})",
                "options": ["A. Direct but incomplete approach", "B. Structured, measurable approach", "C. Over-engineered approach", "D. Irrelevant approach"],
                "answer_key": "B",
                "rationale": "Option B best balances correctness, practicality, and measurable outcomes.",
            }
        return {
            "stem": f"[{context_title}] Explain your approach to solving a realistic {skill.replace('_', ' ')} challenge. (Q{idx})",
            "answer_guidelines": ["Clear reasoning", "Skill-specific application", "Actionable recommendation"],
        }
    
    base_context = module_name if mode == "curriculum" and module_name else context_title
    skill_display = skill.replace('_', ' ')
    
    # Create context-specific prompt based on question type
    if qtype == "mcq":
        prompt = f"""Generate a {difficulty} multiple-choice question about {skill_display} for a {difficulty} learner.
Context: {base_context}

Format the response as JSON with these exact fields:
{{
    "stem": "question text here",
    "options": ["A. option 1", "B. option 2", "C. option 3", "D. option 4"],
    "answer_key": "A",
    "rationale": "explanation of why this is correct"
}}

The question should be practical and applicable to the context: {base_context}.
Difficulty level: {difficulty}"""
    elif qtype == "saq":
        prompt = f"""Generate a {difficulty} short-answer question about {skill_display} for a {difficulty} learner.
Context: {base_context}

Format the response as JSON with these exact fields:
{{
    "stem": "question text here",
    "answer_guidelines": ["guideline 1", "guideline 2", "guideline 3"]
}}

The question should require a thoughtful response demonstrating {skill_display} in {base_context}.
Difficulty level: {difficulty}"""
    elif qtype == "caselet":
        prompt = f"""Generate a {difficulty} caselet/scenario-based question about {skill_display} for a {difficulty} learner.
Context: {base_context}

Format the response as JSON with these exact fields:
{{
    "stem": "scenario description followed by the question",
    "answer_guidelines": ["guideline 1", "guideline 2", "guideline 3"]
}}

The caselet should present a realistic scenario in {base_context} and ask the learner to apply {skill_display}.
Difficulty level: {difficulty}"""
    else:  # coding
        prompt = f"""Generate a {difficulty} coding question about {skill_display} for a {difficulty} learner.
Context: {base_context}

Format the response as JSON with these exact fields:
{{
    "stem": "problem statement and requirements",
    "answer_guidelines": ["Correctness on sample and edge cases", "Time and space complexity explanation", "Readable and maintainable implementation"]
}}

The coding question should test {skill_display} skills in the context of {base_context}.
Difficulty level: {difficulty}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are an expert question generator for technical assessments. Generate high-quality questions that test real competencies. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        
        # Extract the JSON from the response
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            question_data = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                question_data = json.loads(json_match.group())
            else:
                # Fallback to default structure
                question_data = {
                    "stem": f"[{base_context}] Question about {skill_display} (Q{idx})",
                    "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"] if qtype == "mcq" else None,
                    "answer_key": "B" if qtype == "mcq" else None,
                    "answer_guidelines": ["Demonstrate understanding", "Apply concept correctly", "Explain your reasoning"],
                }
        
        return question_data
        
    except Exception as e:
        logger.exception("LLM generation failed; using fallback questions. Error: %s", str(e))
        # Return a fallback question
        if qtype == "mcq":
            return {
                "stem": f"[{base_context}] Which option best demonstrates {skill_display}? (Q{idx})",
                "options": ["A. Direct but incomplete approach", "B. Structured, measurable approach", "C. Over-engineered approach", "D. Irrelevant approach"],
                "answer_key": "B",
                "rationale": "Option B best balances correctness, practicality, and measurable outcomes."
            }
        else:
            return {
                "stem": f"[{base_context}] Explain your approach to solving a realistic {skill_display} challenge. (Q{idx})",
                "answer_guidelines": ["Clear reasoning", "Skill-specific application", "Actionable recommendation"]
            }



def _build_skill_targets(skills: Iterable[str], weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    normalized = [_normalize_skill_name(skill) for skill in skills if skill and skill.strip()]
    if not normalized:
        return {"conceptual_understanding": 1.0}

    if weights:
        merged: Dict[str, float] = {}
        for k, v in weights.items():
            if v <= 0:
                continue
            merged[_normalize_skill_name(k)] = float(v)
        if merged:
            total = sum(merged.values())
            return {k: v / total for k, v in merged.items()}

    count = Counter(normalized)
    total_count = sum(count.values())
    return {k: v / total_count for k, v in count.items()}


def _allocate_count_by_weight(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    if total <= 0:
        return {k: 0 for k in weights}

    raw = {k: total * w for k, w in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(base.values())

    if remainder > 0:
        order = sorted(raw.items(), key=lambda pair: pair[1] - int(pair[1]), reverse=True)
        idx = 0
        while remainder > 0 and order:
            key = order[idx % len(order)][0]
            base[key] += 1
            remainder -= 1
            idx += 1

    return base


def _difficulty_sequence(distribution: Dict[DifficultyLevel, float], total: int) -> List[DifficultyLevel]:
    allocations = _allocate_count_by_weight(total, distribution)
    seq: List[DifficultyLevel] = []
    for level in ("easy", "medium", "hard"):
        seq.extend([level] * allocations.get(level, 0))
    if len(seq) < total:
        seq.extend(["medium"] * (total - len(seq)))
    return seq[:total]


def _extract_jd_skills(
    jd_title: str,
    jd_text: str,
    explicit_skills: Optional[List[str]] = None,
    custom_role_skill_map: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    combined = f"{jd_title}\n{jd_text}".lower()
    found: List[str] = []

    effective_map = DEFAULT_JD_SKILL_MAP.copy()
    if custom_role_skill_map:
        for canonical_skill, hints in custom_role_skill_map.items():
            normalized_key = _normalize_skill_name(canonical_skill)
            normalized_hints = [str(hint).strip().lower() for hint in hints if str(hint).strip()]
            if normalized_hints:
                effective_map[normalized_key] = normalized_hints

    for canonical_skill, hints in effective_map.items():
        if any(hint in combined for hint in hints):
            found.append(canonical_skill)

    for skill in explicit_skills or []:
        normalized = _normalize_skill_name(skill)
        if normalized not in found:
            found.append(normalized)

    return found or ["problem_solving", "communication"]


def extract_jd_topics(jd_title: str, jd_text: str, max_topics: int = 2) -> List[str]:
    """Extract up to max_topics from JD text using OpenAI with deterministic fallback."""
    max_topics = max(1, min(max_topics, 5))
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            prompt = (
                "Extract up to "
                f"{max_topics} core technical/interview topics from this JD. "
                "Return only strict JSON in this schema: "
                '{"topics": ["topic 1", "topic 2"]}. '
                "Keep each topic short (1-3 words) and remove duplicates.\n\n"
                f"Title: {jd_title}\n\nJD:\n{jd_text[:12000]}"
            )

            response = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": "You extract concise interview topics and always return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=180,
            )

            content = (response.choices[0].message.content or "").strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                parsed = json.loads(match.group()) if match else {}

            topics = parsed.get("topics", []) if isinstance(parsed, dict) else []
            cleaned: List[str] = []
            seen = set()
            for topic in topics:
                if not isinstance(topic, str):
                    continue
                t = topic.strip()
                if not t:
                    continue
                key = t.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(t)
                if len(cleaned) >= max_topics:
                    break

            if cleaned:
                return cleaned
        except Exception:
            logger.exception("JD topic extraction via OpenAI failed; using fallback extraction")

    fallback = _extract_jd_skills(jd_title, jd_text)
    fallback_topics: List[str] = []
    for skill in fallback:
        pretty = skill.replace("_", " ").strip().title()
        if pretty and pretty.lower() not in [v.lower() for v in fallback_topics]:
            fallback_topics.append(pretty)
        if len(fallback_topics) >= max_topics:
            break

    return fallback_topics or ["Problem Solving", "Communication"][:max_topics]


def extract_jd_title(jd_text: str, fallback_title: str = "Role Assessment") -> str:
    """Extract a concise JD title from text using OpenAI with deterministic fallback."""
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            prompt = (
                "Extract the most likely job title from this job description. "
                "Return only strict JSON in this schema: "
                '{"title": "..."}. '
                "Keep title concise (2-6 words), no department suffixes, no location.\n\n"
                f"JD:\n{jd_text[:12000]}"
            )

            response = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": "You extract concise role titles and always return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=80,
            )

            content = (response.choices[0].message.content or "").strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                parsed = json.loads(match.group()) if match else {}

            title = parsed.get("title", "") if isinstance(parsed, dict) else ""
            if isinstance(title, str) and title.strip():
                return title.strip()[:80]
        except Exception:
            logger.exception("JD title extraction via OpenAI failed; using fallback extraction")

    # Fallback 1: explicit markers commonly used in JDs
    patterns = [
        r"(?im)^\s*job\s*title\s*[:\-]\s*(.+)$",
        r"(?im)^\s*role\s*[:\-]\s*(.+)$",
        r"(?im)^\s*position\s*[:\-]\s*(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, jd_text)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate[:80]

    # Fallback 2: use the first meaningful line
    lines = [line.strip() for line in jd_text.splitlines() if line.strip()]
    for line in lines[:8]:
        # Skip noisy headings
        if len(line) < 2:
            continue
        if line.lower() in {"job description", "about the role", "responsibilities", "requirements"}:
            continue
        if len(line.split()) <= 8:
            return line[:80]

    return fallback_title


def build_blueprint(payload: Dict[str, Any]) -> Blueprint:
    mode: Literal["curriculum", "jd"] = payload["mode"]
    level = str(payload.get("target_level", "intermediate")).lower()

    difficulty_distribution: Dict[DifficultyLevel, float] = payload.get("difficulty_distribution") or DEFAULT_DIFFICULTY_BY_LEVEL.get(
        level, DEFAULT_DIFFICULTY_BY_LEVEL["intermediate"]
    )

    type_distribution: Dict[QuestionType, int] = payload.get("type_distribution") or DEFAULT_TYPE_SPLIT_BY_MODE[mode]

    if mode == "curriculum":
        curriculum = payload["curriculum"]
        all_skills: List[str] = []
        for module in curriculum.get("modules", []):
            all_skills.extend(module.get("skills", []))
        all_skills.extend(curriculum.get("skills", []))
        skill_targets = _build_skill_targets(all_skills, payload.get("skill_weights"))
    else:
        jd = payload["jd"]
        extracted_skills = _extract_jd_skills(
            jd_title=jd.get("title", ""),
            jd_text=jd.get("description", ""),
            explicit_skills=jd.get("skills", []),
            custom_role_skill_map=payload.get("role_skill_map"),
        )
        skill_targets = _build_skill_targets(extracted_skills, payload.get("skill_weights"))

    return Blueprint(
        mode=mode,
        difficulty_distribution=difficulty_distribution,
        type_distribution=type_distribution,
        skill_targets=skill_targets,
    )


def _build_question_stem(mode: str, qtype: QuestionType, skill: str, context_title: str, module_name: Optional[str], idx: int) -> str:
    base_context = module_name if mode == "curriculum" and module_name else context_title

    if qtype == "mcq":
        return f"[{base_context}] Which option best demonstrates {skill.replace('_', ' ')} in practice? (Q{idx})"
    if qtype == "saq":
        return f"[{base_context}] Explain your approach to solving a realistic {skill.replace('_', ' ')} challenge. (Q{idx})"
    if qtype == "caselet":
        return f"[{base_context}] Analyze this scenario and propose a decision framework focused on {skill.replace('_', ' ')}. (Q{idx})"
    return f"[{base_context}] Write a function or query to solve a {skill.replace('_', ' ')} task with clear complexity considerations. (Q{idx})"


def _map_question_type(qtype: QuestionType) -> str:
    if qtype == "mcq":
        return "objective"
    if qtype == "coding":
        return "coding"
    if qtype == "caselet":
        return "caselet"
    return "subjective"


async def _build_item(
    item_id: str,
    mode: str,
    qtype: QuestionType,
    difficulty: DifficultyLevel,
    skill: str,
    context_title: str,
    module_name: Optional[str],
    idx: int,
) -> Dict[str, Any]:
    # Generate question using LLM
    llm_data = await _generate_question_with_llm(
        mode=mode,
        qtype=qtype,
        difficulty=difficulty,
        skill=skill,
        context_title=context_title,
        module_name=module_name,
        idx=idx,
    )
    
    # Build base item structure
    item: Dict[str, Any] = {
        "item_id": item_id,
        "type": qtype,
        "question_type": _map_question_type(qtype),
        "difficulty": difficulty,
        "difficulty_level": difficulty,
        "skill_tags": [skill],
        "module": module_name,
        "stem": llm_data.get("stem", _build_question_stem(mode, qtype, skill, context_title, module_name, idx)),
        "review_status": "pending",
        "source_trace": {
            "mode": mode,
            "context": context_title,
            "module": module_name,
            "primary_skill": skill,
        },
    }
    
    # Add type-specific fields from LLM data
    if qtype == "mcq":
        item["options"] = llm_data.get("options", [
            "A. Direct but incomplete approach",
            "B. Structured, measurable approach",
            "C. Over-engineered approach",
            "D. Irrelevant approach",
        ])
        item["answer_key"] = llm_data.get("answer_key", "B")
        item["rationale"] = llm_data.get("rationale", "Option B best balances correctness, practicality, and measurable outcomes.")
    elif qtype == "coding":
        item["answer_guidelines"] = llm_data.get("answer_guidelines", [
            "Correctness on sample and edge cases",
            "Time and space complexity explanation",
            "Readable and maintainable implementation",
        ])
    else:
        item["answer_guidelines"] = llm_data.get("answer_guidelines", [
            "Clear reasoning",
            "Skill-specific application",
            "Actionable recommendation",
        ])

    return item


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _semantic_tokens(text: str) -> set[str]:
    synonyms = {
        "analyse": "analyze",
        "analysis": "analyze",
        "optimise": "optimize",
        "optimization": "optimize",
        "kpi": "metric",
        "metrics": "metric",
        "querying": "query",
        "queries": "query",
        "sql": "database",
        "database": "database",
        "scenario": "case",
        "caselet": "case",
        "problem": "challenge",
        "problems": "challenge",
        "stakeholder": "communication",
        "presentation": "communication",
        "roadmap": "planning",
    }

    base = _tokenize(text)
    collapsed = {synonyms.get(token, token) for token in base}
    return {token for token in collapsed if len(token) >= 3}


def _semantic_similarity(a: str, b: str) -> float:
    ta, tb = _semantic_tokens(a), _semantic_tokens(b)
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _semantic_redundancy_pairs(items: List[Dict[str, Any]], threshold: float = 0.68) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            stem_i = items[i].get("stem", "")
            stem_j = items[j].get("stem", "")
            semantic_score = _semantic_similarity(stem_i, stem_j)
            if semantic_score >= threshold:
                lexical_score = _similarity(stem_i, stem_j)
                pairs.append(
                    {
                        "pair": [items[i]["item_id"], items[j]["item_id"]],
                        "semantic_score": round(semantic_score, 4),
                        "lexical_score": round(lexical_score, 4),
                    }
                )
    return pairs


def _redundancy_pairs(items: List[Dict[str, Any]], threshold: float = 0.8) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _similarity(items[i].get("stem", ""), items[j].get("stem", "")) >= threshold:
                pairs.append((items[i]["item_id"], items[j]["item_id"]))
    return pairs


def _item_text_for_scoring(item: Dict[str, Any]) -> str:
    parts: List[str] = [str(item.get("stem", ""))]
    if isinstance(item.get("answer_guidelines"), list):
        parts.extend([str(x) for x in item.get("answer_guidelines", [])])
    if isinstance(item.get("options"), list):
        parts.extend([str(x) for x in item.get("options", [])])
    return " ".join(parts).strip()


def _extract_keywords(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "your", "have", "has", "are", "into", "using",
        "need", "role", "job", "description", "about", "will", "their", "our", "you", "how", "what", "when",
    }
    tokens = _tokenize(text)
    return {token for token in tokens if len(token) >= 3 and token not in stopwords}


def _objective_alignment_report(assessment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    objectives_by_module = assessment.get("learning_objectives_by_module") or {}
    if not isinstance(objectives_by_module, dict) or not objectives_by_module:
        return None

    items = assessment.get("items", [])
    per_item: List[Dict[str, Any]] = []
    low_alignment_items: List[Dict[str, Any]] = []

    module_scores: Dict[str, List[float]] = {}
    total_score = 0.0

    for item in items:
        module_name = (item.get("module") or "").strip()
        module_objectives = objectives_by_module.get(module_name, []) if module_name else []

        if not module_objectives:
            entry = {
                "item_id": item.get("item_id"),
                "module": module_name or None,
                "score": 0.0,
                "matched_objective": None,
            }
            per_item.append(entry)
            low_alignment_items.append(entry)
            continue

        item_tokens = _extract_keywords(_item_text_for_scoring(item))
        best_score = 0.0
        best_objective: Optional[str] = None

        for objective in module_objectives:
            obj_tokens = _extract_keywords(str(objective))
            if not obj_tokens:
                continue
            score = _safe_ratio(len(item_tokens & obj_tokens), len(obj_tokens))
            if score > best_score:
                best_score = score
                best_objective = str(objective)

        rounded_score = round(best_score, 4)
        entry = {
            "item_id": item.get("item_id"),
            "module": module_name,
            "score": rounded_score,
            "matched_objective": best_objective,
        }
        per_item.append(entry)
        total_score += best_score

        if best_score < 0.2:
            low_alignment_items.append(entry)

        module_scores.setdefault(module_name or "General", []).append(best_score)

    avg_score = round(_safe_ratio(int(total_score * 10000), len(items) * 10000), 4) if items else 0.0
    per_module_average = {
        module: round(sum(scores) / len(scores), 4) if scores else 0.0
        for module, scores in module_scores.items()
    }

    return {
        "average_score": avg_score,
        "per_module_average": per_module_average,
        "low_alignment_count": len(low_alignment_items),
        "low_alignment_items": low_alignment_items,
        "per_item": per_item,
    }


def _hiring_relevance_report(assessment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if assessment.get("mode") != "jd":
        return None

    jd_context = assessment.get("jd_context") or {}
    title = str(jd_context.get("title", assessment.get("title", "")))
    jd_skills = [str(skill) for skill in jd_context.get("skills", []) if str(skill).strip()]
    context_terms = set(jd_context.get("keywords", []))
    context_terms |= _extract_keywords(title)

    generic_patterns = [
        "what is",
        "define",
        "explain concept",
        "generic",
        "best practice",
        "in general",
    ]

    items = assessment.get("items", [])
    flagged_items: List[Dict[str, Any]] = []
    per_item: List[Dict[str, Any]] = []
    total_score = 0.0

    for item in items:
        item_text = _item_text_for_scoring(item).lower()
        item_terms = _extract_keywords(item_text)

        tagged_skills = [str(skill) for skill in item.get("skill_tags", []) if str(skill).strip()]
        effective_skills = tagged_skills or jd_skills

        mentions_skill = any(
            _normalize_skill_name(skill).replace("_", " ") in item_text or _normalize_skill_name(skill) in item_text
            for skill in effective_skills
        )

        role_term_overlap = _safe_ratio(len(item_terms & context_terms), len(context_terms)) if context_terms else 0.0
        generic_hit_count = sum(1 for pattern in generic_patterns if pattern in item_text)

        score = 0.0
        if mentions_skill:
            score += 0.5
        score += min(role_term_overlap * 0.5, 0.4)
        if generic_hit_count > 0:
            score -= min(0.1 * generic_hit_count, 0.3)

        score = max(0.0, min(score, 1.0))
        rounded_score = round(score, 4)
        total_score += score

        is_generic = rounded_score < 0.3 or (generic_hit_count >= 2 and not mentions_skill)
        reason = None
        if is_generic:
            if not mentions_skill:
                reason = "No explicit JD skill evidence"
            elif role_term_overlap < 0.1:
                reason = "Weak role-context grounding"
            else:
                reason = "Likely generic phrasing"

        entry = {
            "item_id": item.get("item_id"),
            "score": rounded_score,
            "is_generic_risk": is_generic,
            "reason": reason,
        }
        per_item.append(entry)
        if is_generic:
            flagged_items.append(entry)

    avg_score = round(_safe_ratio(int(total_score * 10000), len(items) * 10000), 4) if items else 0.0

    return {
        "average_score": avg_score,
        "generic_risk_count": len(flagged_items),
        "flagged_items": flagged_items,
        "per_item": per_item,
    }


def _effectiveness_report(assessment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = assessment.get("effectiveness_metrics") or assessment.get("attempt_analytics")
    if not isinstance(source, dict):
        return None

    skill_gap_by_learner = source.get("skill_gap_by_learner") or source.get("learner_skill_gap") or []
    item_pass_rates = source.get("item_pass_rates") or source.get("question_pass_rates") or []
    time_spent_per_item = source.get("time_spent_per_item") or source.get("question_time_spent") or []
    discrimination = source.get("discrimination") or source.get("question_discrimination") or {}

    if not any([skill_gap_by_learner, item_pass_rates, time_spent_per_item, discrimination]):
        return None

    over_discriminating = []
    under_discriminating = []
    if isinstance(discrimination, dict):
        over_discriminating = discrimination.get("over_discriminating") or discrimination.get("high_discrimination") or []
        under_discriminating = discrimination.get("under_discriminating") or discrimination.get("low_discrimination") or []

    summary = source.get("summary") or {}

    return {
        "summary": {
            "attempt_count": summary.get("attempt_count"),
            "candidate_count": summary.get("candidate_count"),
            "learner_count": summary.get("learner_count"),
        },
        "skill_gap_by_learner": skill_gap_by_learner,
        "item_pass_rates": item_pass_rates,
        "time_spent_per_item": time_spent_per_item,
        "discrimination": {
            "over_discriminating": over_discriminating,
            "under_discriminating": under_discriminating,
        },
    }


async def generate_assessment(payload: Dict[str, Any]) -> Dict[str, Any]:
    blueprint = build_blueprint(payload)

    total_questions = sum(blueprint.type_distribution.values())
    skill_alloc = _allocate_count_by_weight(total_questions, blueprint.skill_targets)

    context_title = ""
    module_rotation: List[Optional[str]] = []
    learning_objectives_by_module: Dict[str, List[str]] = {}
    if blueprint.mode == "curriculum":
        curriculum = payload["curriculum"]
        context_title = curriculum.get("course", "Curriculum Assessment")
        modules = curriculum.get("modules", [])
        learning_objectives_by_module = {
            str(module.get("name", "")).strip(): [str(obj) for obj in module.get("learning_objectives", []) if str(obj).strip()]
            for module in modules
            if str(module.get("name", "")).strip()
        }
        if modules:
            module_names = [m.get("name", "General") for m in modules]
            module_rotation = [module_names[i % len(module_names)] for i in range(total_questions)]
        else:
            module_rotation = [None] * total_questions
    else:
        jd = payload["jd"]
        context_title = jd.get("title", "Role Assessment")
        module_rotation = [None] * total_questions

    difficulty_seq = _difficulty_sequence(blueprint.difficulty_distribution, total_questions)

    skills_flat: List[str] = []
    for skill, count in skill_alloc.items():
        skills_flat.extend([skill] * count)
    if len(skills_flat) < total_questions:
        skills_flat.extend([list(blueprint.skill_targets.keys())[0]] * (total_questions - len(skills_flat)))
    skills_flat = skills_flat[:total_questions]

    items: List[Dict[str, Any]] = []
    running_index = 1
    pointer = 0

    for qtype, count in blueprint.type_distribution.items():
        for _ in range(count):
            skill = skills_flat[pointer]
            difficulty = difficulty_seq[pointer]
            module_name = module_rotation[pointer]
            item_id = f"Q{running_index:03d}"

            item = await _build_item(
                item_id=item_id,
                mode=blueprint.mode,
                qtype=qtype,
                difficulty=difficulty,
                skill=skill,
                context_title=context_title,
                module_name=module_name,
                idx=running_index,
            )
            items.append(item)
            pointer += 1
            running_index += 1

    # Keep only unique stems to reduce repetition and then top up if needed.
    unique_items: List[Dict[str, Any]] = []
    seen_stems: set[str] = set()
    for item in items:
        if item["stem"] in seen_stems:
            continue
        seen_stems.add(item["stem"])
        unique_items.append(item)

    if len(unique_items) < total_questions:
        for idx in range(len(unique_items), total_questions):
            template = unique_items[idx % len(unique_items)] if unique_items else items[0]
            clone = dict(template)
            clone["item_id"] = f"Q{idx + 1:03d}"
            clone["stem"] = f"{template['stem']} (variant {idx + 1})"
            unique_items.append(clone)

    assessment = {
        "mode": blueprint.mode,
        "title": context_title,
        "difficulty_distribution": blueprint.difficulty_distribution,
        "skill_targets": blueprint.skill_targets,
        "type_distribution": blueprint.type_distribution,
        "items": unique_items,
    }

    if learning_objectives_by_module:
        assessment["learning_objectives_by_module"] = learning_objectives_by_module

    if blueprint.mode == "jd":
        jd = payload.get("jd", {})
        jd_title = str(jd.get("title", context_title))
        jd_description = str(jd.get("description", ""))
        explicit_skills = [str(skill) for skill in jd.get("skills", []) if str(skill).strip()]
        derived_keywords = list(_extract_keywords(f"{jd_title} {jd_description}"))[:40]
        assessment["jd_context"] = {
            "title": jd_title,
            "skills": explicit_skills,
            "keywords": derived_keywords,
        }

    return assessment


def build_coverage_report(assessment: Dict[str, Any]) -> Dict[str, Any]:
    items = assessment.get("items", [])
    total = len(items)

    skill_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for item in items:
        for skill in item.get("skill_tags", []):
            skill_counts[skill] += 1
        difficulty_counts[item.get("difficulty", "medium")] += 1
        type_counts[item.get("type", "mcq")] += 1

    targets = assessment.get("skill_targets", {})
    achieved = {skill: _safe_ratio(count, total) for skill, count in skill_counts.items()}

    target_vs_achieved = {}
    for skill, target in targets.items():
        actual = achieved.get(skill, 0.0)
        target_vs_achieved[skill] = {
            "target": round(target, 4),
            "achieved": round(actual, 4),
            "gap": round(target - actual, 4),
        }

    redundancy = _redundancy_pairs(items)
    semantic_redundancy = _semantic_redundancy_pairs(items)

    accepted = sum(1 for item in items if item.get("review_status") == "accepted")
    rejected = sum(1 for item in items if item.get("review_status") == "rejected")

    report = {
        "question_count": total,
        "skill_coverage": {
            "target_vs_achieved": target_vs_achieved,
            "uncovered_skills": [s for s, data in target_vs_achieved.items() if data["achieved"] == 0.0],
        },
        "difficulty_coverage": {
            "actual_distribution": {
                level: round(_safe_ratio(difficulty_counts.get(level, 0), total), 4)
                for level in ("easy", "medium", "hard")
            }
        },
        "question_type_coverage": dict(type_counts),
        "redundancy": {
            "duplicate_pair_count": len(redundancy),
            "duplicate_pairs": redundancy,
            "semantic_duplicate_pair_count": len(semantic_redundancy),
            "semantic_duplicate_pairs": semantic_redundancy,
        },
        "review_summary": {
            "accepted": accepted,
            "rejected": rejected,
            "pending": total - accepted - rejected,
        },
    }

    objective_alignment = _objective_alignment_report(assessment)
    if objective_alignment is not None:
        report["objective_alignment"] = objective_alignment

    hiring_relevance = _hiring_relevance_report(assessment)
    if hiring_relevance is not None:
        report["hiring_relevance"] = hiring_relevance

    effectiveness_report = _effectiveness_report(assessment)
    if effectiveness_report is not None:
        report["effectiveness_report"] = effectiveness_report

    return report


def apply_review_actions(assessment: Dict[str, Any], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    items_by_id = {item["item_id"]: item for item in assessment.get("items", [])}

    for action in actions:
        item_id = action["item_id"]
        decision = action["action"]
        if item_id not in items_by_id:
            continue

        item = items_by_id[item_id]
        if decision == "accept":
            item["review_status"] = "accepted"
        elif decision == "reject":
            item["review_status"] = "rejected"
            if action.get("reason"):
                item["rejection_reason"] = action["reason"]
        elif decision == "edit":
            edited = action.get("edited_item", {})
            for key, value in edited.items():
                if key == "item_id":
                    continue
                item[key] = value
            item["review_status"] = "edited"

    return assessment

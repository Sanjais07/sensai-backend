from fastapi import APIRouter
from typing import List, Dict, Any
from api.db.task import (
    get_all_scorecards_for_org as get_all_scorecards_for_org_from_db,
    update_scorecard as update_scorecard_from_db,
    create_scorecard as create_scorecard_from_db,
)
from api.models import Scorecard, BaseScorecard, CreateScorecardRequest
from pydantic import BaseModel
import json
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

router = APIRouter()


class ScorecardRecommendRequest(BaseModel):
    question_text: str
    question_title: str | None = None
    scorecards: List[Dict[str, Any]]


@router.get("/", response_model=List[Scorecard])
async def get_all_scorecards_for_org(org_id: int) -> List[Scorecard]:
    return await get_all_scorecards_for_org_from_db(org_id)


@router.put("/{scorecard_id}")
async def update_scorecard(scorecard_id: int, scorecard: BaseScorecard) -> Scorecard:
    return await update_scorecard_from_db(scorecard_id, scorecard)


@router.post("/", response_model=Scorecard)
async def create_scorecard(scorecard: CreateScorecardRequest) -> Scorecard:
    return await create_scorecard_from_db(scorecard.model_dump())


@router.post("/ai/recommend")
async def recommend_scorecard_with_ai(
    payload: ScorecardRecommendRequest,
) -> Dict[str, Any]:
    """
    Use OpenAI to recommend the best existing scorecard for a question.
    Returns the recommended scorecard ID and data.
    """
    if not OpenAI:
        return {
            "success": False,
            "error": "OpenAI not installed"
        }
    
    question_text = payload.question_text
    question_title = payload.question_title
    scorecards = payload.scorecards

    if not scorecards:
        return {
            "success": False,
            "error": "No scorecards available"
        }
    
    try:
        client = OpenAI()
        
        # Format scorecards for the prompt
        scorecard_descriptions = "\n".join([
            f"Scorecard {i+1} (ID: {sc.get('id')}): {sc.get('title') or sc.get('name') or 'Unnamed'}\n"
            f"  Description: {sc.get('description', 'No description')}\n"
            f"  Criteria: {json.dumps(sc.get('criteria', []), indent=2) if sc.get('criteria') else 'N/A'}"
            for i, sc in enumerate(scorecards)
        ])
        
        prompt = f"""You are an expert educator. Analyze this question and recommend the BEST matching scorecard from the available options.

Question Title: {question_title or 'Assessment'}
Question: {question_text}

Available Scorecards:
{scorecard_descriptions}

Respond with ONLY a JSON object (no markdown, no code blocks):
{{
  "recommended_scorecard_id": <the ID of the best matching scorecard>,
  "reason": "brief explanation why this scorecard is the best match",
  "confidence": 0.85
}}"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert educator. Analyze and recommend the best scorecard. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            result = json.loads(response_text)
        
        # Find the recommended scorecard
        recommended_id = result.get("recommended_scorecard_id")
        recommended_scorecard = next(
            (sc for sc in scorecards if sc.get('id') == recommended_id),
            None
        )
        
        if not recommended_scorecard:
            return {
                "success": False,
                "error": f"Recommended scorecard ID {recommended_id} not found"
            }
        
        return {
            "success": True,
            "scorecard_id": recommended_id,
            "scorecard_data": recommended_scorecard,
            "reason": result.get("reason", ""),
            "confidence": float(result.get("confidence", 0.8) or 0.8)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

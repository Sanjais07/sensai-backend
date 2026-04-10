from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.assessment_engine import (
    apply_review_actions,
    build_coverage_report,
    generate_assessment,
)
from api.db.assessment import (
    create_assessment,
    get_assessment_by_id,
    update_assessment,
    save_assessment_review,
    list_assessments,
)

router = APIRouter()


class CurriculumModuleInput(BaseModel):
    name: str
    skills: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)


class CurriculumInput(BaseModel):
    course: str
    modules: List[CurriculumModuleInput] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)


class JDInput(BaseModel):
    title: str
    description: str
    skills: List[str] = Field(default_factory=list)


class AssessmentGenerateRequest(BaseModel):
    mode: Literal["curriculum", "jd"]
    target_level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    curriculum: Optional[CurriculumInput] = None
    jd: Optional[JDInput] = None
    type_distribution: Optional[Dict[Literal["mcq", "saq", "caselet", "coding"], int]] = None
    difficulty_distribution: Optional[Dict[Literal["easy", "medium", "hard"], float]] = None
    skill_weights: Optional[Dict[str, float]] = None


class AssessmentGenerateResponse(BaseModel):
    assessment: Dict[str, Any]
    coverage_report: Dict[str, Any]


class ReviewAction(BaseModel):
    item_id: str
    action: Literal["accept", "edit", "reject"]
    reason: Optional[str] = None
    edited_item: Optional[Dict[str, Any]] = None


class ReviewRequest(BaseModel):
    assessment: Dict[str, Any]
    actions: List[ReviewAction]


class ValidateRequest(BaseModel):
    assessment: Dict[str, Any]


class SaveAssessmentRequest(BaseModel):
    org_id: int
    mode: str
    title: str
    assessment_json: Dict[str, Any]
    cohort_id: Optional[int] = None
    user_id: Optional[int] = None


class SaveAssessmentResponse(BaseModel):
    id: int
    org_id: int
    mode: str
    title: str
    created_at: str


class SaveReviewRequest(BaseModel):
    assessment_id: int
    user_id: int
    review_actions: List[Dict[str, Any]]
    coverage_report: Dict[str, Any]


@router.post("/generate", response_model=AssessmentGenerateResponse)
async def generate_assessment_endpoint(payload: AssessmentGenerateRequest):
    data = payload.model_dump(exclude_none=True)

    if payload.mode == "curriculum" and not payload.curriculum:
        return {
            "assessment": {},
            "coverage_report": {
                "error": "curriculum is required when mode=curriculum",
            },
        }

    if payload.mode == "jd" and not payload.jd:
        return {
            "assessment": {},
            "coverage_report": {
                "error": "jd is required when mode=jd",
            },
        }

    assessment = await generate_assessment(data)
    coverage_report = build_coverage_report(assessment)
    return {
        "assessment": assessment,
        "coverage_report": coverage_report,
    }


@router.post("/validate")
async def validate_assessment(payload: ValidateRequest):
    return {
        "coverage_report": build_coverage_report(payload.assessment),
    }


@router.post("/review")
async def review_assessment(payload: ReviewRequest):
    updated_assessment = apply_review_actions(
        assessment=payload.assessment,
        actions=[action.model_dump(exclude_none=True) for action in payload.actions],
    )
    return {
        "assessment": updated_assessment,
        "coverage_report": build_coverage_report(updated_assessment),
    }


@router.post("/save", response_model=SaveAssessmentResponse)
async def save_assessment_endpoint(payload: SaveAssessmentRequest):
    """Save an assessment to the database."""
    assessment_id = await create_assessment(
        org_id=payload.org_id,
        mode=payload.mode,
        title=payload.title,
        assessment_json=payload.assessment_json,
        cohort_id=payload.cohort_id,
        user_id=payload.user_id,
    )
    
    assessment = await get_assessment_by_id(assessment_id)
    if not assessment:
        raise HTTPException(status_code=500, detail="Failed to save assessment")
    
    return {
        "id": assessment["id"],
        "org_id": assessment["org_id"],
        "mode": assessment["mode"],
        "title": assessment["title"],
        "created_at": assessment["created_at"],
    }


@router.get("/{assessment_id}")
async def get_assessment_endpoint(assessment_id: int):
    """Get an assessment by ID."""
    assessment = await get_assessment_by_id(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment


@router.get("")
async def list_assessments_endpoint(org_id: int, mode: Optional[str] = None, cohort_id: Optional[int] = None, limit: int = 50, offset: int = 0):
    """List assessments with pagination."""
    assessments, total_count = await list_assessments(
        org_id=org_id,
        mode=mode,
        cohort_id=cohort_id,
        limit=limit,
        offset=offset,
    )
    return {
        "assessments": assessments,
        "total": total_count,
        "limit": limit,
        "offset": offset,
    }


@router.post("/{assessment_id}/save-review")
async def save_assessment_review_endpoint(assessment_id: int, payload: SaveReviewRequest):
    """Save a review for an assessment."""
    # Verify assessment exists
    assessment = await get_assessment_by_id(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    review_id = await save_assessment_review(
        assessment_id=assessment_id,
        user_id=payload.user_id,
        review_actions=payload.review_actions,
        coverage_report=payload.coverage_report,
    )
    
    return {
        "review_id": review_id,
        "assessment_id": assessment_id,
        "message": "Review saved successfully",
    }


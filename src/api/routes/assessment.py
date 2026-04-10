from io import BytesIO
from typing import Any, Dict, List, Literal, Optional
import importlib
import zipfile
import xml.etree.ElementTree as ET

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.assessment_engine import (
    apply_review_actions,
    build_coverage_report,
    extract_jd_title,
    extract_jd_topics,
    generate_assessment,
)
from api.db.assessment import (
    create_assessment,
    get_assessment_by_id,
    update_assessment,
    save_assessment_review,
    list_assessments,
    create_assessment_task_in_course,
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
    course_id: Optional[int] = None
    course_ids: Optional[List[int]] = None


def _extract_text_from_docx_bytes(content: bytes) -> str:
    with zipfile.ZipFile(BytesIO(content)) as archive:
        try:
            xml_data = archive.read("word/document.xml")
        except KeyError:
            return ""

    root = ET.fromstring(xml_data)
    texts = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            texts.append(node.text)
    return " ".join(texts)


def _extract_text_from_upload(filename: str, content_type: str, content: bytes) -> str:
    lower_name = (filename or "").lower()
    lower_type = (content_type or "").lower()

    is_pdf = lower_name.endswith(".pdf") or "pdf" in lower_type
    is_docx = lower_name.endswith(".docx") or (
        "wordprocessingml.document" in lower_type
    )
    is_doc = lower_name.endswith(".doc") or "msword" in lower_type

    if is_pdf:
        pypdf_module = importlib.import_module("pypdf")
        pdf_reader = getattr(pypdf_module, "PdfReader")
        reader = pdf_reader(BytesIO(content))
        page_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(page_text).strip()

    if is_docx:
        return _extract_text_from_docx_bytes(content).strip()

    if is_doc:
        # Legacy .doc is binary; this best-effort decode extracts any plain text fragments.
        return content.decode("latin-1", errors="ignore").strip()

    raise HTTPException(status_code=400, detail="Only PDF or Word files are supported")


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


@router.post("/jd-topics")
async def extract_jd_topics_endpoint(
    file: UploadFile = File(...),
    jd_title: str = Form("Role Assessment"),
):
    """Extract text from uploaded JD file and return one topic."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 20MB")

    extracted_text = _extract_text_from_upload(file.filename or "", file.content_type or "", content)
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Unable to extract text from the uploaded file")

    topics = extract_jd_topics(jd_title=jd_title, jd_text=extracted_text, max_topics=1)
    extracted_title = extract_jd_title(jd_text=extracted_text, fallback_title=jd_title or "Role Assessment")

    return {
        "topics": topics[:1],
        "jd_title": extracted_title,
        "text_length": len(extracted_text),
        "extracted_text": extracted_text,
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

    created_task_ids: List[int] = []
    requested_course_ids: List[int] = []
    if payload.course_ids:
        requested_course_ids.extend(payload.course_ids)
    if payload.course_id:
        requested_course_ids.append(payload.course_id)

    unique_course_ids = list(dict.fromkeys([cid for cid in requested_course_ids if cid]))

    for course_id in unique_course_ids:
        created_task_id = await create_assessment_task_in_course(
            course_id=course_id,
            assessment_title=assessment.get("title", "Generated Assessment"),
            assessment_json=assessment.get("assessment_json", {}),
        )
        if created_task_id:
            created_task_ids.append(created_task_id)
    
    return {
        "review_id": review_id,
        "assessment_id": assessment_id,
        "created_task_id": created_task_ids[0] if created_task_ids else None,
        "created_task_ids": created_task_ids,
        "message": "Review saved successfully",
    }


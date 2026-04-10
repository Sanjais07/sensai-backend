import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from api.config import assessments_table_name, assessment_reviews_table_name, course_milestones_table_name
from api.utils.db import get_new_db_connection, execute_db_operation
from api.db.task import create_draft_task_for_course, update_draft_quiz
from api.models import TaskType, TaskStatus


async def create_assessment(
    org_id: int,
    mode: str,
    title: str,
    assessment_json: Dict[str, Any],
    cohort_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> int:
    """Create a new assessment record in database."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        
        assessment_json_str = json.dumps(assessment_json)
        
        await cursor.execute(
            f"""
            INSERT INTO {assessments_table_name} 
            (org_id, cohort_id, user_id, mode, title, assessment_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                org_id,
                cohort_id,
                user_id,
                mode,
                title,
                assessment_json_str,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
            ),
        )
        
        assessment_id = cursor.lastrowid
        await conn.commit()
        
        return assessment_id


async def get_assessment_by_id(assessment_id: int) -> Optional[Dict[str, Any]]:
    """Fetch assessment by ID."""
    row = await execute_db_operation(
        f"SELECT id, org_id, cohort_id, user_id, mode, title, assessment_json, created_at, updated_at FROM {assessments_table_name} WHERE id = ?",
        (assessment_id,),
        fetch_one=True,
    )
    
    if not row:
        return None
    
    assessment_json = json.loads(row[6]) if row[6] else {}
    
    return {
        "id": row[0],
        "org_id": row[1],
        "cohort_id": row[2],
        "user_id": row[3],
        "mode": row[4],
        "title": row[5],
        "assessment_json": assessment_json,
        "created_at": row[7],
        "updated_at": row[8],
    }


async def update_assessment(
    assessment_id: int, 
    assessment_json: Dict[str, Any],
) -> bool:
    """Update assessment JSON."""
    assessment_json_str = json.dumps(assessment_json)
    
    await execute_db_operation(
        f"""
        UPDATE {assessments_table_name}
        SET assessment_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (assessment_json_str, datetime.utcnow().isoformat(), assessment_id),
    )
    
    return True


async def save_assessment_review(
    assessment_id: int,
    user_id: int,
    review_actions: List[Dict[str, Any]],
    coverage_report: Dict[str, Any],
) -> int:
    """Save review decision for an assessment."""
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        
        review_actions_str = json.dumps(review_actions)
        coverage_report_str = json.dumps(coverage_report)
        
        await cursor.execute(
            f"""
            INSERT INTO {assessment_reviews_table_name}
            (assessment_id, user_id, review_actions, coverage_report, reviewed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                assessment_id,
                user_id,
                review_actions_str,
                coverage_report_str,
                datetime.utcnow().isoformat(),
            ),
        )
        
        review_id = cursor.lastrowid
        await conn.commit()
        
        return review_id


async def get_assessment_reviews(assessment_id: int) -> List[Dict[str, Any]]:
    """Get all reviews for an assessment."""
    rows = await execute_db_operation(
        f"""
        SELECT id, assessment_id, user_id, review_actions, coverage_report, reviewed_at 
        FROM {assessment_reviews_table_name}
        WHERE assessment_id = ?
        ORDER BY reviewed_at DESC
        """,
        (assessment_id,),
        fetch_all=True,
    )
    
    reviews = []
    for row in rows:
        reviews.append({
            "id": row[0],
            "assessment_id": row[1],
            "user_id": row[2],
            "review_actions": json.loads(row[3]) if row[3] else [],
            "coverage_report": json.loads(row[4]) if row[4] else {},
            "reviewed_at": row[5],
        })
    
    return reviews


async def list_assessments(
    org_id: int,
    mode: Optional[str] = None,
    cohort_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """List assessments with pagination."""
    query = f"SELECT id, org_id, cohort_id, user_id, mode, title, assessment_json, created_at, updated_at FROM {assessments_table_name} WHERE org_id = ?"
    params: List[Any] = [org_id]
    
    if mode:
        query += " AND mode = ?"
        params.append(mode)
    
    if cohort_id:
        query += " AND cohort_id = ?"
        params.append(cohort_id)
    
    # Get total count
    count_query = f"SELECT COUNT(*) FROM {assessments_table_name} WHERE org_id = ?"
    count_params: List[Any] = [org_id]
    
    if mode:
        count_query += " AND mode = ?"
        count_params.append(mode)
    
    if cohort_id:
        count_query += " AND cohort_id = ?"
        count_params.append(cohort_id)
    
    count_row = await execute_db_operation(
        count_query,
        tuple(count_params),
        fetch_one=True,
    )
    total_count = count_row[0] if count_row else 0
    
    # Get paginated results
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    rows = await execute_db_operation(
        query,
        tuple(params),
        fetch_all=True,
    )
    
    assessments = []
    for row in rows:
        assessment_json = json.loads(row[6]) if row[6] else {}
        assessments.append({
            "id": row[0],
            "org_id": row[1],
            "cohort_id": row[2],
            "user_id": row[3],
            "mode": row[4],
            "title": row[5],
            "assessment_json": assessment_json,
            "created_at": row[7],
            "updated_at": row[8],
        })
    
    return assessments, total_count


def _to_richtext_blocks(text: str) -> List[Dict[str, Any]]:
    return [
        {
            "type": "paragraph",
            "props": {},
            "content": [{"type": "text", "text": text}],
            "children": [],
        }
    ]


def _assessment_item_to_quiz_question(item: Dict[str, Any]) -> Dict[str, Any]:
    stem = str(item.get("stem") or "")
    item_id = str(item.get("item_id") or "")
    item_type = str(item.get("type") or "mcq").strip().lower()
    item_question_type = str(item.get("question_type") or "").strip().lower()
    options = item.get("options") or []
    answer_key = str(item.get("answer_key") or "")

    answer_text = answer_key
    if options and answer_key:
        matching = [option for option in options if option.strip().lower().startswith(answer_key.strip().lower())]
        if matching:
            answer_text = matching[0]

    is_coding_question = item_type in {"coding", "code"} or item_question_type in {"coding", "code"}

    question_type = "objective" if item_type == "mcq" else "subjective"
    input_type = "code" if is_coding_question else "text"
    coding_languages = ["python"] if is_coding_question else None

    return {
        "blocks": _to_richtext_blocks(stem),
        "answer": _to_richtext_blocks(answer_text) if answer_text else None,
        "type": question_type,
        "input_type": input_type,
        "response_type": "chat",
        "context": {
            "source": "assessment_engine",
            "item_id": item_id,
            "skill_tags": item.get("skill_tags", []),
            "difficulty": item.get("difficulty"),
        },
        "coding_languages": coding_languages,
        "scorecard_id": None,
        "title": item_id or "Assessment Question",
        "settings": {
            "allowCopyPaste": True,
            "generatedBy": "assessment_engine",
            "options": options,
            "questionType": question_type,
        },
        "max_attempts": None,
        "is_feedback_shown": True,
    }


async def create_assessment_task_in_course(course_id: int, assessment_title: str, assessment_json: Dict[str, Any]) -> Optional[int]:
    """Create a draft quiz task in the first module of a course for the saved assessment."""
    milestone_row = await execute_db_operation(
        f"""
        SELECT milestone_id
        FROM {course_milestones_table_name}
        WHERE course_id = ? AND deleted_at IS NULL
        ORDER BY ordering ASC
        LIMIT 1
        """,
        (course_id,),
        fetch_one=True,
    )

    if not milestone_row or milestone_row[0] is None:
        return None

    task_title = f"Assessment: {assessment_title}"
    task_id, _ = await create_draft_task_for_course(
        title=task_title,
        type=str(TaskType.QUIZ),
        course_id=course_id,
        milestone_id=milestone_row[0],
    )

    items = assessment_json.get("items", []) if isinstance(assessment_json, dict) else []
    questions = [_assessment_item_to_quiz_question(item) for item in items if isinstance(item, dict)]

    if questions:
        await update_draft_quiz(
            task_id=task_id,
            title=task_title,
            questions=questions,
            scheduled_publish_at=None,
            status=TaskStatus.DRAFT,
        )

    return task_id

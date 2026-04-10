from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException

from api.config import chat_history_table_name
from api.db.analytics import get_cohort_completion
from api.db.cohort import (
    get_all_cohorts_for_org,
    get_cohort_attempt_data_for_tasks,
    get_cohort_by_id,
)
from api.db.course import get_courses_for_cohort
from api.db.user import get_user_organizations
from api.utils.db import execute_db_operation

from api.insights_engine import (
    CohortSignal,
    FeedbackSignal,
    InsightAnalysisResponse,
    InsightWorkspaceRequest,
    LearnerSignal,
    ModuleSignal,
    generate_insight_analysis,
    get_sample_workspace_payload,
)

router = APIRouter()


@router.get("/sample", response_model=InsightWorkspaceRequest)
def get_sample_workspace() -> InsightWorkspaceRequest:
    return get_sample_workspace_payload()


@router.post("/analyze", response_model=InsightAnalysisResponse)
def analyze_insights(
    workspace: Optional[InsightWorkspaceRequest] = None,
) -> InsightAnalysisResponse:
    payload = workspace or get_sample_workspace_payload()
    return generate_insight_analysis(payload)


@router.get("/live/users/{user_id}/organizations")
async def get_live_user_organizations(user_id: int) -> List[Dict]:
    organizations = await get_user_organizations(user_id)
    return [
        {
            "id": int(org["id"]),
            "name": org["name"],
            "role": org.get("role"),
        }
        for org in organizations
    ]


@router.get("/live/organizations/{org_id}/cohorts")
async def get_live_org_cohorts(org_id: int) -> List[Dict]:
    cohorts = await get_all_cohorts_for_org(org_id)
    return [
        {
            "id": int(cohort["id"]),
            "name": cohort["name"],
            "org_id": org_id,
        }
        for cohort in cohorts
    ]


@router.get("/live/cohorts/{cohort_id}/courses")
async def get_live_cohort_courses(cohort_id: int) -> List[Dict]:
    cohort = await get_cohort_by_id(cohort_id)
    if not cohort:
        raise HTTPException(status_code=404, detail="Cohort not found")

    courses = await get_courses_for_cohort(cohort_id, include_tree=False)
    return [
        {
            "id": int(course["id"]),
            "name": course["name"],
        }
        for course in courses
    ]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _parse_sqlite_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None

    normalized = value.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


async def _build_live_workspace_from_cohort(
    cohort_id: int,
    course_id: Optional[int] = None,
    batch_id: Optional[int] = None,
) -> InsightWorkspaceRequest:
    cohort = await get_cohort_by_id(cohort_id, batch_id)
    if not cohort:
        raise HTTPException(status_code=404, detail="Cohort not found")

    members = cohort.get("members", [])
    learners = [member for member in members if member.get("role") == "learner"]
    learner_ids = [int(member["id"]) for member in learners]

    courses = await get_courses_for_cohort(cohort_id, include_tree=True)
    if course_id is not None:
        courses = [course for course in courses if int(course.get("id", 0)) == int(course_id)]

    task_to_module: Dict[int, str] = {}
    task_to_title: Dict[int, str] = {}
    module_to_tasks: Dict[str, List[int]] = defaultdict(list)

    for course in courses:
        course_name = course.get("name", "Course")
        for milestone in course.get("milestones", []):
            milestone_name = milestone.get("name") or "Module"
            module_name = f"{course_name} - {milestone_name}"
            for task in milestone.get("tasks", []):
                task_id = int(task.get("id"))
                task_to_module[task_id] = module_name
                task_to_title[task_id] = task.get("title", f"Task {task_id}")
                module_to_tasks[module_name].append(task_id)

    all_task_ids = sorted(task_to_module.keys())

    completion_data = (
        await get_cohort_completion(cohort_id, learner_ids, course_id)
        if learner_ids
        else {}
    )

    attempt_data = (
        await get_cohort_attempt_data_for_tasks(cohort_id, all_task_ids, batch_id)
        if learner_ids and all_task_ids
        else []
    )

    attempt_map: Dict[int, Dict[int, int]] = defaultdict(dict)
    for row in attempt_data:
        user_id = int(row["user_id"])
        for task_id in all_task_ids:
            attempt_map[user_id][task_id] = int(row.get(f"task_{task_id}", 0))

    user_ids_str = ",".join(map(str, learner_ids)) if learner_ids else ""
    task_ids_str = ",".join(map(str, all_task_ids)) if all_task_ids else ""

    message_count_map: Dict[int, Dict[int, int]] = defaultdict(dict)
    last_active_map: Dict[int, Optional[datetime]] = {user_id: None for user_id in learner_ids}
    if learner_ids and all_task_ids:
        message_rows = await execute_db_operation(
            f"""
            SELECT user_id, task_id, COUNT(*) as message_count
            FROM {chat_history_table_name}
            WHERE user_id IN ({user_ids_str})
              AND task_id IN ({task_ids_str})
              AND deleted_at IS NULL
            GROUP BY user_id, task_id
            """,
            fetch_all=True,
        )

        for user_id, task_id, message_count in message_rows:
            message_count_map[int(user_id)][int(task_id)] = int(message_count)

        last_active_rows = await execute_db_operation(
            f"""
            SELECT user_id, MAX(created_at)
            FROM {chat_history_table_name}
            WHERE user_id IN ({user_ids_str})
              AND task_id IN ({task_ids_str})
              AND deleted_at IS NULL
            GROUP BY user_id
            """,
            fetch_all=True,
        )

        for user_id, last_active in last_active_rows:
            last_active_map[int(user_id)] = _parse_sqlite_datetime(last_active)

    now_utc = datetime.now(timezone.utc)

    learner_signals: List[LearnerSignal] = []
    unresolved_feedback: List[FeedbackSignal] = []

    for learner in learners:
        user_id = int(learner["id"])
        learner_name = (
            f"{learner.get('first_name') or ''} {learner.get('last_name') or ''}".strip()
            or learner.get("email")
            or f"Learner {user_id}"
        )

        user_completion = completion_data.get(user_id, {})

        lowest_module_name = "General"
        lowest_module_completion = 1.0
        selected_retries = 0
        selected_time_spent = 0
        selected_error_signature: Optional[str] = None

        for module_name, module_task_ids in module_to_tasks.items():
            if not module_task_ids:
                continue

            module_completed_count = 0
            module_message_count = 0
            unresolved_task_titles: List[str] = []

            for task_id in module_task_ids:
                task_completion = user_completion.get(task_id, {"is_complete": False})
                if task_completion.get("is_complete"):
                    module_completed_count += 1
                else:
                    unresolved_task_titles.append(task_to_title.get(task_id, f"Task {task_id}"))

                module_message_count += int(message_count_map.get(user_id, {}).get(task_id, 0))

            module_completion = _safe_ratio(module_completed_count, len(module_task_ids))
            module_retries = max(0, module_message_count - len(module_task_ids))
            module_time_spent = module_message_count * 2

            if module_completion <= lowest_module_completion:
                lowest_module_completion = module_completion
                lowest_module_name = module_name
                selected_retries = module_retries
                selected_time_spent = module_time_spent
                selected_error_signature = unresolved_task_titles[0] if unresolved_task_titles else None

        last_active = last_active_map.get(user_id)
        if last_active is None:
            last_active_days = 999
        else:
            delta = now_utc - last_active.astimezone(timezone.utc)
            last_active_days = max(0, delta.days)

        learner_signals.append(
            LearnerSignal(
                learner_id=str(user_id),
                learner_name=learner_name,
                module_name=lowest_module_name,
                retries=selected_retries,
                average_score=round(lowest_module_completion, 2),
                time_spent_minutes=selected_time_spent,
                last_active_days=last_active_days,
                error_signature=selected_error_signature,
                feedback_tags=["live-data"],
            )
        )

        if selected_error_signature and lowest_module_completion < 0.65:
            unresolved_feedback.append(
                FeedbackSignal(
                    source="ai",
                    scope="module",
                    target_id=lowest_module_name,
                    sentiment="negative",
                    text=f"{learner_name} is repeatedly blocked by '{selected_error_signature}'.",
                )
            )

    module_signals: List[ModuleSignal] = []
    for module_name, module_task_ids in module_to_tasks.items():
        if not module_task_ids or not learner_ids:
            continue

        learner_completion_rates: List[float] = []
        unresolved_titles_counter: Dict[str, int] = defaultdict(int)
        affected_learners = 0

        for learner in learners:
            user_id = int(learner["id"])
            user_completion = completion_data.get(user_id, {})
            completed_count = 0
            for task_id in module_task_ids:
                task_completion = user_completion.get(task_id, {"is_complete": False})
                if task_completion.get("is_complete"):
                    completed_count += 1
                else:
                    unresolved_titles_counter[task_to_title.get(task_id, f"Task {task_id}")] += 1

            completion_rate = _safe_ratio(completed_count, len(module_task_ids))
            learner_completion_rates.append(completion_rate)
            if completion_rate < 0.65:
                affected_learners += 1

        avg_completion = _safe_ratio(sum(learner_completion_rates), len(learner_completion_rates))
        failure_rate = 1.0 - avg_completion

        sorted_signatures = sorted(
            unresolved_titles_counter.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        module_signals.append(
            ModuleSignal(
                module_id=module_name.lower().replace(" ", "-"),
                module_name=module_name,
                completion_rate=round(avg_completion, 2),
                failure_rate=round(max(0.0, failure_rate), 2),
                affected_learners=affected_learners,
                repeated_error_signatures=[item[0] for item in sorted_signatures[:3]],
                learning_objectives=[],
            )
        )

    current_window_count = 0
    previous_window_count = 0
    if learner_ids and all_task_ids:
        engagement_rows = await execute_db_operation(
            f"""
            SELECT
              SUM(CASE WHEN datetime(created_at) >= datetime('now', '-7 days') THEN 1 ELSE 0 END) as current_count,
              SUM(CASE WHEN datetime(created_at) < datetime('now', '-7 days') AND datetime(created_at) >= datetime('now', '-14 days') THEN 1 ELSE 0 END) as previous_count
            FROM {chat_history_table_name}
            WHERE user_id IN ({user_ids_str})
              AND task_id IN ({task_ids_str})
              AND deleted_at IS NULL
            """,
            fetch_one=True,
        )
        if engagement_rows:
            current_window_count = int(engagement_rows[0] or 0)
            previous_window_count = int(engagement_rows[1] or 0)

    if previous_window_count > 0:
        wow_change = (current_window_count - previous_window_count) / previous_window_count
    else:
        wow_change = 0.0

    active_last_7 = sum(
        1
        for user_id in learner_ids
        if last_active_map.get(user_id)
        and (now_utc - last_active_map[user_id].astimezone(timezone.utc)).days <= 7
    )
    inactive_over_14 = sum(
        1
        for user_id in learner_ids
        if last_active_map.get(user_id) is None
        or (now_utc - last_active_map[user_id].astimezone(timezone.utc)).days > 14
    )

    attendance_rate = _safe_ratio(active_last_7, len(learner_ids)) if learner_ids else 0.0
    drop_rate = _safe_ratio(inactive_over_14, len(learner_ids)) if learner_ids else 0.0

    cohort_name = cohort.get("name") or f"Cohort {cohort_id}"
    cohort_signal = CohortSignal(
        cohort_id=str(cohort_id),
        cohort_name=cohort_name,
        week_over_week_engagement_change=round(wow_change, 2),
        attendance_rate=round(attendance_rate, 2),
        active_learner_rate=round(attendance_rate, 2),
        drop_rate=round(drop_rate, 2),
    )

    workspace_name = f"{cohort_name} Live Intelligence"
    return InsightWorkspaceRequest(
        workspace_name=workspace_name,
        personas=["mentor", "creator", "operator"],
        learners=learner_signals,
        modules=module_signals,
        cohorts=[cohort_signal],
        feedback=unresolved_feedback,
    )


@router.post("/live/cohorts/{cohort_id}/analyze", response_model=InsightAnalysisResponse)
async def analyze_live_cohort_insights(
    cohort_id: int,
    course_id: Optional[int] = None,
    batch_id: Optional[int] = None,
) -> InsightAnalysisResponse:
    workspace = await _build_live_workspace_from_cohort(
        cohort_id=cohort_id,
        course_id=course_id,
        batch_id=batch_id,
    )
    return generate_insight_analysis(workspace)

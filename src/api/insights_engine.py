from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

PersonaType = Literal["mentor", "creator", "operator"]
PriorityType = Literal["high", "medium", "low"]
ScopeType = Literal["learner", "module", "cohort", "cross-cutting"]
ExecutionType = Literal["manual", "suggested", "auto"]
SignalType = Literal["individual", "systemic", "mixed"]
SentimentType = Literal["positive", "neutral", "negative"]


class LearnerSignal(BaseModel):
    learner_id: str
    learner_name: str
    module_name: str
    retries: int = 0
    average_score: float = 0.0
    time_spent_minutes: int = 0
    last_active_days: int = 0
    error_signature: Optional[str] = None
    feedback_tags: List[str] = Field(default_factory=list)


class ModuleSignal(BaseModel):
    module_id: str
    module_name: str
    completion_rate: float = 0.0
    failure_rate: float = 0.0
    affected_learners: int = 0
    repeated_error_signatures: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)


class CohortSignal(BaseModel):
    cohort_id: str
    cohort_name: str
    week_over_week_engagement_change: float = 0.0
    attendance_rate: float = 0.0
    active_learner_rate: float = 0.0
    drop_rate: float = 0.0


class FeedbackSignal(BaseModel):
    source: Literal["ai", "human"]
    scope: ScopeType
    target_id: str
    sentiment: SentimentType = "neutral"
    text: str


class InsightWorkspaceRequest(BaseModel):
    workspace_name: str = "SensAI Intelligence Workspace"
    personas: List[PersonaType] = Field(default_factory=lambda: ["mentor", "creator", "operator"])
    learners: List[LearnerSignal] = Field(default_factory=list)
    modules: List[ModuleSignal] = Field(default_factory=list)
    cohorts: List[CohortSignal] = Field(default_factory=list)
    feedback: List[FeedbackSignal] = Field(default_factory=list)


class InsightAction(BaseModel):
    label: str
    rationale: str
    execution: ExecutionType = "suggested"
    urgency: PriorityType = "medium"
    target_scope: ScopeType = "cross-cutting"


class PersonaInsight(BaseModel):
    persona: PersonaType
    title: str
    summary: str
    priority: PriorityType
    scope: ScopeType
    signal_type: SignalType
    confidence: float
    evidence: List[str] = Field(default_factory=list)
    impacted_entities: List[str] = Field(default_factory=list)
    recommended_actions: List[InsightAction] = Field(default_factory=list)
    auto_trigger: bool = False


class ValidationReport(BaseModel):
    individual_signal_count: int
    systemic_signal_count: int
    redundancy_score: float
    noise_score: float
    coverage_gaps: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class InsightAnalysisResponse(BaseModel):
    workspace_name: str
    generated_at: datetime
    summary: str
    insights: List[PersonaInsight]
    validation: ValidationReport
    export_json: Dict[str, Any]


def _priority_from_severity(severity: float) -> PriorityType:
    if severity >= 0.75:
        return "high"
    if severity >= 0.45:
        return "medium"
    return "low"


def _confidence_from_signals(signal_count: int, severity: float) -> float:
    confidence = 0.55 + min(0.35, signal_count * 0.08) + min(0.1, severity * 0.1)
    return round(min(confidence, 0.97), 2)


def _action(label: str, rationale: str, execution: ExecutionType = "suggested", urgency: PriorityType = "medium", target_scope: ScopeType = "cross-cutting") -> InsightAction:
    return InsightAction(
        label=label,
        rationale=rationale,
        execution=execution,
        urgency=urgency,
        target_scope=target_scope,
    )


def _sample_workspace() -> InsightWorkspaceRequest:
    return InsightWorkspaceRequest(
        workspace_name="DP Cohort Intelligence Demo",
        learners=[
            LearnerSignal(
                learner_id="learner-101",
                learner_name="Aarav",
                module_name="Dynamic Programming Module 2",
                retries=4,
                average_score=0.42,
                time_spent_minutes=118,
                last_active_days=1,
                error_signature="state transition confusion",
                feedback_tags=["stuck", "needs practice"],
            ),
            LearnerSignal(
                learner_id="learner-102",
                learner_name="Meera",
                module_name="Dynamic Programming Module 2",
                retries=3,
                average_score=0.51,
                time_spent_minutes=104,
                last_active_days=2,
                error_signature="base-case miss",
                feedback_tags=["needs walkthrough"],
            ),
            LearnerSignal(
                learner_id="learner-103",
                learner_name="Rohan",
                module_name="Trees Module 1",
                retries=1,
                average_score=0.79,
                time_spent_minutes=62,
                last_active_days=0,
                feedback_tags=["good progress"],
            ),
            LearnerSignal(
                learner_id="learner-104",
                learner_name="Zoya",
                module_name="Dynamic Programming Module 2",
                retries=2,
                average_score=0.47,
                time_spent_minutes=91,
                last_active_days=6,
                error_signature="overlapping subproblems gap",
                feedback_tags=["slow", "needs hint"],
            ),
        ],
        modules=[
            ModuleSignal(
                module_id="module-dp-2",
                module_name="Dynamic Programming Module 2",
                completion_rate=0.58,
                failure_rate=0.41,
                affected_learners=3,
                repeated_error_signatures=["state transition confusion", "base-case miss", "overlapping subproblems gap"],
                learning_objectives=["identify states", "build transitions", "reason about complexity"],
            ),
            ModuleSignal(
                module_id="module-trees-1",
                module_name="Trees Module 1",
                completion_rate=0.84,
                failure_rate=0.14,
                affected_learners=1,
                repeated_error_signatures=["recursive traversal ordering"],
                learning_objectives=["tree traversal patterns", "recursion"],
            ),
        ],
        cohorts=[
            CohortSignal(
                cohort_id="cohort-1",
                cohort_name="Spring 2026 Cohort",
                week_over_week_engagement_change=-0.2,
                attendance_rate=0.74,
                active_learner_rate=0.69,
                drop_rate=0.12,
            )
        ],
        feedback=[
            FeedbackSignal(
                source="ai",
                scope="module",
                target_id="module-dp-2",
                sentiment="negative",
                text="Learners repeatedly miss the state transition and need alternate explanations.",
            ),
            FeedbackSignal(
                source="human",
                scope="cohort",
                target_id="cohort-1",
                sentiment="negative",
                text="Attendance feels inconsistent and learners are asking for a live revision session.",
            ),
        ],
    )


def _collect_negative_feedback(feedback: List[FeedbackSignal], target_scope: ScopeType) -> List[FeedbackSignal]:
    return [item for item in feedback if item.scope == target_scope and item.sentiment == "negative"]


def _summarize_module_feedback(module_name: str, feedback: List[FeedbackSignal]) -> List[str]:
    if not feedback:
        return []
    unique_texts = []
    for item in feedback:
        text = item.text.strip()
        if text and text not in unique_texts:
            unique_texts.append(text)
    return [f"{module_name}: {text}" for text in unique_texts[:2]]


def _mentor_insights(workspace: InsightWorkspaceRequest) -> List[PersonaInsight]:
    learner_groups: Dict[str, List[LearnerSignal]] = defaultdict(list)
    for learner in workspace.learners:
        if learner.average_score <= 0.68 or learner.retries >= 2 or learner.last_active_days >= 5:
            learner_groups[learner.module_name].append(learner)

    insights: List[PersonaInsight] = []

    for module_name, learners in learner_groups.items():
        severity = max(
            1.0 - mean([learner.average_score for learner in learners]),
            min(1.0, max(learner.retries for learner in learners) / 4),
        )
        priority = _priority_from_severity(severity)
        confidence = _confidence_from_signals(len(learners), severity)
        evidence = [
            f"{learner.learner_name} has {learner.retries} retries and {learner.average_score:.0%} average score"
            for learner in learners[:3]
        ]
        if any(learner.error_signature for learner in learners):
            evidence.extend(
                [
                    f"Repeated error pattern: {learner.error_signature}"
                    for learner in learners
                    if learner.error_signature
                ][:2]
            )

        insights.append(
            PersonaInsight(
                persona="mentor",
                title=f"{len(learners)} learners need outreach in {module_name}",
                summary=f"These learners are showing repeat attempts, lower scores, or recent inactivity in {module_name}.",
                priority=priority,
                scope="learner",
                signal_type="individual",
                confidence=confidence,
                evidence=evidence,
                impacted_entities=[learner.learner_name for learner in learners],
                recommended_actions=[
                    _action(
                        "Trigger 1:1 outreach",
                        "The learners are personally stuck, so a short mentor check-in will unblock them quickly.",
                        execution="manual",
                        urgency=priority,
                        target_scope="learner",
                    ),
                    _action(
                        "Assign targeted practice set",
                        "A focused practice bundle can reinforce the exact concepts they are missing.",
                        execution="suggested",
                        urgency="medium",
                        target_scope="learner",
                    ),
                ],
                auto_trigger=priority == "high" and confidence >= 0.8,
            )
        )

    return insights


def _creator_insights(workspace: InsightWorkspaceRequest) -> List[PersonaInsight]:
    insights: List[PersonaInsight] = []
    negative_feedback_by_module: Dict[str, List[FeedbackSignal]] = defaultdict(list)
    for item in workspace.feedback:
        if item.scope == "module" and item.sentiment == "negative":
            negative_feedback_by_module[item.target_id].append(item)

    for module in workspace.modules:
        feedback_items = negative_feedback_by_module.get(module.module_id, [])
        systemic_pressure = max(
            1.0 - module.completion_rate,
            module.failure_rate,
            min(1.0, module.affected_learners / 10),
        )

        if systemic_pressure >= 0.25 or feedback_items:
            priority = _priority_from_severity(systemic_pressure)
            confidence = _confidence_from_signals(max(module.affected_learners, len(feedback_items)), systemic_pressure)
            evidence = [
                f"Completion rate is {module.completion_rate:.0%} and failure rate is {module.failure_rate:.0%}",
                f"{module.affected_learners} learners affected",
            ]
            evidence.extend(
                _summarize_module_feedback(module.module_name, feedback_items)
            )
            if module.repeated_error_signatures:
                evidence.append(
                    f"Common error signatures: {', '.join(module.repeated_error_signatures[:3])}"
                )

            insights.append(
                PersonaInsight(
                    persona="creator",
                    title=f"{module.module_name} needs content revision",
                    summary=f"The module shows a repeat misunderstanding pattern that points to a content or sequencing gap.",
                    priority=priority,
                    scope="module",
                    signal_type="systemic",
                    confidence=confidence,
                    evidence=evidence,
                    impacted_entities=[module.module_name],
                    recommended_actions=[
                        _action(
                            "Revise explanation or add alternate content",
                            "A clearer explanation or alternate representation can reduce the recurring confusion.",
                            execution="manual",
                            urgency=priority,
                            target_scope="module",
                        ),
                        _action(
                            "Insert prerequisite reinforcement",
                            "Learners likely need a smaller prerequisite bridge before this topic.",
                            execution="suggested",
                            urgency="medium",
                            target_scope="module",
                        ),
                    ],
                    auto_trigger=priority == "high" and confidence >= 0.8,
                )
            )

    return insights


def _operator_insights(workspace: InsightWorkspaceRequest) -> List[PersonaInsight]:
    insights: List[PersonaInsight] = []
    for cohort in workspace.cohorts:
        engagement_pressure = max(
            abs(min(0.0, cohort.week_over_week_engagement_change)),
            max(0.0, 0.8 - cohort.attendance_rate),
            cohort.drop_rate,
        )

        if engagement_pressure >= 0.18:
            priority = _priority_from_severity(engagement_pressure)
            confidence = _confidence_from_signals(1, engagement_pressure)
            evidence = [
                f"Week-over-week engagement changed by {cohort.week_over_week_engagement_change:.0%}",
                f"Attendance rate is {cohort.attendance_rate:.0%}",
                f"Active learner rate is {cohort.active_learner_rate:.0%}",
            ]
            if cohort.drop_rate:
                evidence.append(f"Drop rate is {cohort.drop_rate:.0%}")

            insights.append(
                PersonaInsight(
                    persona="operator",
                    title=f"{cohort.cohort_name} needs a cohort-wide intervention",
                    summary="Cohort engagement is slipping enough that a program-level response will likely be more effective than isolated outreach.",
                    priority=priority,
                    scope="cohort",
                    signal_type="systemic",
                    confidence=confidence,
                    evidence=evidence,
                    impacted_entities=[cohort.cohort_name],
                    recommended_actions=[
                        _action(
                            "Launch live intervention session",
                            "A live cohort session can reset momentum and address blockers at scale.",
                            execution="manual",
                            urgency=priority,
                            target_scope="cohort",
                        ),
                        _action(
                            "Send a cohort challenge or nudge",
                            "A short challenge or reminder can rebuild weekly participation.",
                            execution="suggested",
                            urgency="medium",
                            target_scope="cohort",
                        ),
                    ],
                    auto_trigger=priority == "high" and confidence >= 0.78,
                )
            )

    return insights


def generate_insight_analysis(workspace: InsightWorkspaceRequest) -> InsightAnalysisResponse:
    payload = workspace if any([workspace.learners, workspace.modules, workspace.cohorts, workspace.feedback]) else _sample_workspace()

    insights: List[PersonaInsight] = []
    if "mentor" in payload.personas:
        insights.extend(_mentor_insights(payload))
    if "creator" in payload.personas:
        insights.extend(_creator_insights(payload))
    if "operator" in payload.personas:
        insights.extend(_operator_insights(payload))

    individual_signal_count = sum(1 for insight in insights if insight.signal_type == "individual")
    systemic_signal_count = sum(1 for insight in insights if insight.signal_type == "systemic")

    all_evidence = [item for insight in insights for item in insight.evidence]
    unique_evidence = len(set(all_evidence))
    redundancy_score = 0.0 if not all_evidence else round(1.0 - (unique_evidence / len(all_evidence)), 2)

    low_confidence = [insight for insight in insights if insight.confidence < 0.75]
    noise_score = round(min(1.0, (len(low_confidence) / max(len(insights), 1)) + redundancy_score * 0.25), 2)

    coverage_gaps: List[str] = []
    if not payload.learners:
        coverage_gaps.append("No learner-level data provided, so individual risk detection is limited.")
    if not payload.modules:
        coverage_gaps.append("No module-level analytics provided, so content gaps cannot be isolated.")
    if not payload.cohorts:
        coverage_gaps.append("No cohort-wide signals provided, so operator-level prioritization is narrower.")
    if not payload.feedback:
        coverage_gaps.append("No feedback logs provided, so explanation quality and learner sentiment are inferred only from performance data.")

    notes = [
        "Insights are sorted by persona relevance and priority.",
        "High-priority, high-confidence items are marked for potential auto-triggering.",
    ]
    if redundancy_score > 0.35:
        notes.append("Signal overlap is elevated, so review repeated evidence before publishing broad actions.")

    summary = (
        f"Generated {len(insights)} actionable insights across {', '.join(sorted(set(payload.personas)))} personas. "
        f"The engine separated {individual_signal_count} individual signals from {systemic_signal_count} systemic signals."
    )

    response = InsightAnalysisResponse(
        workspace_name=payload.workspace_name,
        generated_at=datetime.now(timezone.utc),
        summary=summary,
        insights=insights,
        validation=ValidationReport(
            individual_signal_count=individual_signal_count,
            systemic_signal_count=systemic_signal_count,
            redundancy_score=redundancy_score,
            noise_score=noise_score,
            coverage_gaps=coverage_gaps,
            notes=notes,
        ),
        export_json={},
    )

    response.export_json = response.model_dump(mode="json")
    return response


def get_sample_workspace_payload() -> InsightWorkspaceRequest:
    return _sample_workspace()

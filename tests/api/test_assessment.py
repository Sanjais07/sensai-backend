from fastapi import status


def test_generate_curriculum_assessment(client):
    payload = {
        "mode": "curriculum",
        "target_level": "intermediate",
        "curriculum": {
            "course": "Data Structures",
            "modules": [
                {
                    "name": "Arrays",
                    "skills": ["Problem Solving", "Complexity Analysis"],
                    "learning_objectives": [
                        "Understand indexing and traversal",
                        "Optimize array-based algorithms",
                    ],
                },
                {
                    "name": "Trees",
                    "skills": ["Recursion", "Complexity Analysis"],
                    "learning_objectives": [
                        "Build and traverse trees",
                    ],
                },
            ],
            "skills": ["Conceptual Understanding"],
        },
        "type_distribution": {"mcq": 5, "saq": 2, "coding": 2, "caselet": 1},
    }

    response = client.post("/assessments/generate", json=payload)
    assert response.status_code == status.HTTP_200_OK

    body = response.json()
    assert "assessment" in body
    assert body["assessment"]["mode"] == "curriculum"
    assert len(body["assessment"]["items"]) == 10
    assert "coverage_report" in body
    assert body["coverage_report"]["question_count"] == 10


def test_generate_jd_assessment(client):
    payload = {
        "mode": "jd",
        "target_level": "intermediate",
        "jd": {
            "title": "Product Analyst",
            "description": "Looking for SQL, metrics design, and product thinking to drive experiments.",
            "skills": ["SQL", "Metrics", "Product Thinking"],
        },
    }

    response = client.post("/assessments/generate", json=payload)
    assert response.status_code == status.HTTP_200_OK

    body = response.json()
    assert body["assessment"]["mode"] == "jd"
    assert len(body["assessment"]["items"]) == 21

    skill_keys = body["assessment"]["skill_targets"].keys()
    assert "sql" in skill_keys


def test_review_assessment(client):
    generate_payload = {
        "mode": "jd",
        "jd": {
            "title": "Data Analyst",
            "description": "Need SQL and communication skills",
            "skills": ["SQL", "Communication"],
        },
        "type_distribution": {"mcq": 2, "saq": 1, "caselet": 0},
    }

    generated = client.post("/assessments/generate", json=generate_payload).json()
    assessment = generated["assessment"]

    item_1 = assessment["items"][0]["item_id"]
    item_2 = assessment["items"][1]["item_id"]

    review_payload = {
        "assessment": assessment,
        "actions": [
            {"item_id": item_1, "action": "accept"},
            {
                "item_id": item_2,
                "action": "edit",
                "edited_item": {"stem": "Edited stem for reviewer validation"},
            },
        ],
    }

    response = client.post("/assessments/review", json=review_payload)
    assert response.status_code == status.HTTP_200_OK

    reviewed = response.json()["assessment"]
    items_by_id = {item["item_id"]: item for item in reviewed["items"]}

    assert items_by_id[item_1]["review_status"] == "accepted"
    assert items_by_id[item_2]["review_status"] == "edited"
    assert items_by_id[item_2]["stem"] == "Edited stem for reviewer validation"


def test_validate_assessment(client):
    payload = {
        "assessment": {
            "skill_targets": {"sql": 0.5, "metrics": 0.5},
            "items": [
                {
                    "item_id": "Q001",
                    "type": "mcq",
                    "difficulty": "easy",
                    "skill_tags": ["sql"],
                    "stem": "Question 1",
                    "review_status": "pending",
                },
                {
                    "item_id": "Q002",
                    "type": "saq",
                    "difficulty": "medium",
                    "skill_tags": ["metrics"],
                    "stem": "Question 2",
                    "review_status": "pending",
                },
            ],
        }
    }

    response = client.post("/assessments/validate", json=payload)
    assert response.status_code == status.HTTP_200_OK

    report = response.json()["coverage_report"]
    assert report["question_count"] == 2
    assert "skill_coverage" in report

from fastapi.testclient import TestClient


def test_sample_workspace_endpoint(client: TestClient):
    response = client.get("/insights/sample")

    assert response.status_code == 200
    body = response.json()
    assert body["workspace_name"] == "DP Cohort Intelligence Demo"
    assert len(body["learners"]) > 0
    assert len(body["modules"]) > 0


def test_analyze_insights_endpoint(client: TestClient):
    response = client.post("/insights/analyze", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["workspace_name"] == "DP Cohort Intelligence Demo"
    assert len(body["insights"]) >= 3
    assert body["validation"]["systemic_signal_count"] >= 1
    assert body["validation"]["individual_signal_count"] >= 1
    assert body["export_json"]["workspace_name"] == "DP Cohort Intelligence Demo"


def test_live_cohort_insights_not_found(client: TestClient):
    response = client.post("/insights/live/cohorts/999999/analyze")

    assert response.status_code == 404
    assert response.json()["detail"] == "Cohort not found"
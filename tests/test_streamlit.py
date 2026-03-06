from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


@patch("requests.get")
def test_streamlit_app_loads(mock_get: MagicMock) -> None:
    """Test that the Streamlit app loads without errors.

    Args:
        mock_get (MagicMock): The mock object for requests.get.
    """
    # Mock the collections list response to avoid network calls/timeouts
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    at = AppTest.from_file("docint/app.py")
    at.run(timeout=30)
    assert not at.exception


@patch("requests.post")
@patch("requests.get")
def test_streamlit_dashboard_renders_with_ner_stats(
    mock_get: MagicMock, mock_post: MagicMock
) -> None:
    """Dashboard should render when NER stats endpoint returns data."""

    def _get_side_effect(url: str, *args, **kwargs):
        response = MagicMock()
        response.status_code = 200
        if url.endswith("/collections/list"):
            response.json.return_value = ["alpha"]
        elif url.endswith("/sessions/list"):
            response.json.return_value = {"sessions": []}
        elif url.endswith("/collections/documents"):
            response.json.return_value = {
                "documents": [
                    {
                        "filename": "doc.pdf",
                        "mimetype": "application/pdf",
                        "node_count": 3,
                    }
                ]
            }
        elif url.endswith("/collections/ner/stats"):
            response.json.return_value = {
                "totals": {
                    "unique_entities": 2,
                    "entity_mentions": 6,
                    "unique_relations": 1,
                },
                "top_entities": [
                    {"text": "Acme", "type": "ORG", "mentions": 4},
                    {"text": "Rivertown", "type": "LOC", "mentions": 2},
                ],
                "entity_types": [
                    {"type": "ORG", "mentions": 4, "unique_entities": 1},
                    {"type": "LOC", "mentions": 2, "unique_entities": 1},
                ],
                "top_relations": [],
                "documents": [],
            }
        else:
            response.json.return_value = {}
        return response

    mock_get.side_effect = _get_side_effect
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})

    at = AppTest.from_file("docint/app.py")
    at.session_state["current_page"] = "Dashboard"
    at.session_state["selected_collection"] = "alpha"
    at.session_state["_cached_collections"] = ["alpha"]
    at.session_state["_cached_sessions"] = []
    at.session_state["_backend_online"] = True
    at.run(timeout=30)
    assert not at.exception

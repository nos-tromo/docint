"""Integration tests for the Streamlit UI pages."""

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


def _analysis_get_side_effect(url: str, *args, **kwargs) -> MagicMock:
    """Return canned HTTP responses for analysis page tests.

    Args:
        url (str): The URL being requested.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        MagicMock: A mock response object with the appropriate JSON data based on the URL.
    """
    _ = args, kwargs
    response = MagicMock()
    response.status_code = 200
    response.text = ""
    if url.endswith("/collections/list"):
        response.json.return_value = ["alpha"]
    elif url.endswith("/sessions/list"):
        response.json.return_value = {"sessions": []}
    elif url.endswith("/collections/ner"):
        response.json.return_value = {
            "sources": [
                {
                    "filename": "doc.pdf",
                    "page": 1,
                    "chunk_id": "n1",
                    "chunk_text": "Acme in Rivertown",
                    "entities": [{"text": "Acme", "type": "ORG"}],
                    "relations": [],
                }
            ]
        }
    elif url.endswith("/collections/ner/graph"):
        response.json.return_value = {
            "nodes": [{"id": "acme::org", "text": "Acme", "mentions": 1}],
            "edges": [],
            "meta": {"node_count": 1, "edge_count": 0},
        }
    elif url.endswith("/collections/hate-speech"):
        response.json.return_value = {"results": []}
    else:
        response.json.return_value = {}
    return response


@patch("requests.post")
@patch("requests.get")
def test_streamlit_analysis_auto_loads_ner_without_summary_call(
    mock_get: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Analysis page should auto-load NER without triggering summary generation.

    Args:
        mock_get (MagicMock): Mocked requests.get function.
        mock_post (MagicMock): Mocked requests.post function.
    """
    mock_get.side_effect = _analysis_get_side_effect

    at = AppTest.from_file("docint/app.py")
    at.session_state["current_page"] = "Analysis"
    at.session_state["selected_collection"] = "alpha"
    at.session_state["_cached_collections"] = ["alpha"]
    at.session_state["_cached_sessions"] = []
    at.session_state["_backend_online"] = True
    at.run(timeout=30)
    assert not at.exception
    assert mock_post.call_count == 0

    ner_calls = [
        call
        for call in mock_get.call_args_list
        if call.args and str(call.args[0]).endswith("/collections/ner")
    ]
    assert len(ner_calls) == 1
    assert ner_calls[0].kwargs.get("params") == {"refresh": "true"}

    at.run(timeout=30)
    ner_calls_after = [
        call
        for call in mock_get.call_args_list
        if call.args and str(call.args[0]).endswith("/collections/ner")
    ]
    assert len(ner_calls_after) == 1
    assert mock_post.call_count == 0


@patch("requests.post")
@patch("requests.get")
def test_streamlit_analysis_manual_summary_generation(
    mock_get: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Summary should be generated only when the user clicks the summary button.

    Args:
        mock_get (MagicMock): Mocked requests.get function.
        mock_post (MagicMock): Mocked requests.post function.
    """
    mock_get.side_effect = _analysis_get_side_effect
    stream_response = MagicMock()
    stream_response.status_code = 200
    stream_response.text = ""
    stream_response.iter_lines.return_value = [
        b'data: {"token":"Summary text."}',
        b'data: {"sources":[{"filename":"doc.pdf","text":"Acme in Rivertown"}]}',
    ]
    mock_post.return_value = stream_response

    at = AppTest.from_file("docint/app.py")
    at.session_state["current_page"] = "Analysis"
    at.session_state["selected_collection"] = "alpha"
    at.session_state["_cached_collections"] = ["alpha"]
    at.session_state["_cached_sessions"] = []
    at.session_state["_backend_online"] = True
    at.run(timeout=30)
    assert mock_post.call_count == 0

    generate_buttons = [btn for btn in at.button if btn.label == "Generate summary"]
    assert generate_buttons
    generate_buttons[0].click()
    at.run(timeout=30)

    assert mock_post.call_count == 1
    summary_state = at.session_state["analysis_summary_by_collection"]["alpha"]
    assert summary_state["generated"] is True
    assert "Summary text." in summary_state["summary"]


@patch("requests.post")
@patch("requests.get")
def test_streamlit_analysis_manual_ner_refresh_refetches(
    mock_get: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Manual NER refresh should trigger a second NER fetch.

    Args:
        mock_get (MagicMock): Mocked requests.get function.
        mock_post (MagicMock): Mocked requests.post function.
    """
    mock_get.side_effect = _analysis_get_side_effect

    at = AppTest.from_file("docint/app.py")
    at.session_state["current_page"] = "Analysis"
    at.session_state["selected_collection"] = "alpha"
    at.session_state["_cached_collections"] = ["alpha"]
    at.session_state["_cached_sessions"] = []
    at.session_state["_backend_online"] = True
    at.run(timeout=30)

    refresh_buttons = [
        btn for btn in at.button if btn.label == "🔄 Refresh NER analysis"
    ]
    assert refresh_buttons
    refresh_buttons[0].click()
    at.run(timeout=30)

    ner_calls = [
        call
        for call in mock_get.call_args_list
        if call.args and str(call.args[0]).endswith("/collections/ner")
    ]
    assert len(ner_calls) == 2
    assert ner_calls[0].kwargs.get("params") == {"refresh": "true"}
    assert ner_calls[1].kwargs.get("params") == {"refresh": "true"}
    assert mock_post.call_count == 0

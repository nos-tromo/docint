from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest


@patch("requests.get")
def test_streamlit_app_loads(mock_get: MagicMock) -> None:
    """
    Test that the Streamlit app loads without errors.

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

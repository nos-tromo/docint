from docint.ui.analysis import _update_summary_metadata


def test_update_summary_metadata_tracks_diagnostics_and_validation() -> None:
    """Analysis metadata updater should persist validation and diagnostics values."""
    result = {
        "validation_checked": None,
        "validation_mismatch": None,
        "validation_reason": None,
        "summary_diagnostics": None,
    }
    event = {
        "validation_checked": True,
        "validation_mismatch": False,
        "validation_reason": None,
        "summary_diagnostics": {
            "total_documents": 5,
            "covered_documents": 4,
            "coverage_ratio": 0.8,
            "coverage_target": 0.7,
            "uncovered_documents": ["doc5.pdf"],
        },
    }

    _update_summary_metadata(result, event)

    assert result["validation_checked"] is True
    assert result["validation_mismatch"] is False
    assert result["validation_reason"] is None
    assert result["summary_diagnostics"] == event["summary_diagnostics"]

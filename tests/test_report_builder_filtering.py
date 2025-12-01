#!/usr/bin/env python3
"""
Tests for report_builder operation_id filtering logic.
- Current implementation FILTERS by operation_id for per-operation reports
- Memories with matching operation_id are included
- Memories with different operation_id are EXCLUDED
- Memories WITHOUT operation_id (untagged) are included for backward compatibility
"""

from unittest.mock import patch

from modules.tools.report_builder import build_report_sections


@patch("modules.tools.report_builder.Mem0ServiceClient")
def test_report_builder_filters_by_operation_id(mock_client_cls):
    """Report builder should filter evidence by operation_id for per-operation reports."""
    op_id = "OP_123"
    # Mock list_memories to return both tagged and untagged
    mock_client = mock_client_cls.return_value
    mock_client.list_memories.return_value = {
        "results": [
            {
                "id": "1",
                "memory": "[VULNERABILITY] A [WHERE] /a",
                "metadata": {"category": "finding", "operation_id": op_id},
            },
            {
                "id": "2",
                "memory": "[VULNERABILITY] B [WHERE] /b",
                "metadata": {"category": "finding", "operation_id": "OP_OTHER"},
            },
            {
                "id": "3",
                "memory": "[VULNERABILITY] C [WHERE] /c",
                "metadata": {"category": "finding"},
            },
        ]
    }

    out = build_report_sections(
        operation_id=op_id, target="example.com", objective="test"
    )
    # Evidence from current operation should be included
    assert any(
        "/a" in e.get("content", "") for e in out.get("raw_evidence", []) or []
    ), "Expected matching evidence from current operation"
    # Evidence from OTHER operations should be EXCLUDED (filtered out)
    assert not any(
        "/b" in e.get("content", "") for e in out.get("raw_evidence", []) or []
    ), "Should EXCLUDE evidence from other operations"
    # Untagged evidence (no operation_id) should be included for backward compatibility
    assert any(
        "/c" in e.get("content", "") for e in out.get("raw_evidence", []) or []
    ), "Should include untagged evidence for backward compatibility"


@patch("modules.tools.report_builder.Mem0ServiceClient")
def test_report_builder_includes_untagged_evidence(mock_client_cls):
    op_id = "OP_456"
    mock_client = mock_client_cls.return_value
    mock_client.list_memories.return_value = {
        "results": [
            {
                "id": "10",
                "memory": "[VULNERABILITY] Legacy [WHERE] /legacy",
                "metadata": {"category": "finding"},
            },
        ]
    }

    out = build_report_sections(
        operation_id=op_id, target="example.com", objective="test"
    )
    # Current implementation includes all evidence (no filtering by operation_id)
    assert out.get("raw_evidence"), "Untagged evidence should be included in the report"
    assert any(
        "/legacy" in e.get("content", "") for e in out.get("raw_evidence", []) or []
    ), "Should include untagged evidence"


@patch("modules.tools.report_builder.Mem0ServiceClient")
def test_report_builder_handles_memory_errors(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.list_memories.side_effect = RuntimeError("boom")

    out = build_report_sections(
        operation_id="OP_ERR", target="example.com", objective="test"
    )
    assert isinstance(out, dict)
    assert out.get("raw_evidence") == [], (
        "Failures loading memories should yield empty evidence rather than crash"
    )

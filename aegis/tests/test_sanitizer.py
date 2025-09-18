"""
Tests for input sanitization functionality.
"""
import pytest
from typing import Dict, Any

from ..sanitizer import InputSanitizer

def test_normalize():
    """Test text normalization."""
    # Basic normalization
    assert InputSanitizer.normalize("hello   world") == "hello world"
    assert InputSanitizer.normalize("hello\r\nworld") == "hello\nworld"
    assert InputSanitizer.normalize("  hello  ") == "hello"
    
    # Unicode normalization
    assert InputSanitizer.normalize("café") == "café"  # NFKC form
    
    # Error handling
    with pytest.raises(ValueError):
        InputSanitizer.normalize(None)  # type: ignore
        
def test_audit_basic():
    """Test basic input auditing."""
    # Valid input
    result = InputSanitizer.audit_text("Hello world")
    assert result["safe"] is True
    assert not result["issues"]
    assert not result["warnings"]
    assert result["normalized"] == "Hello world"
    
    # Invalid type
    result = InputSanitizer.audit_text(None)  # type: ignore
    assert result["safe"] is False
    assert "invalid_type" in result["issues"]
    
def test_critical_issues():
    """Test detection of critical security issues."""
    # Test too long
    long_text = "a" * (InputSanitizer.MAX_INPUT_LENGTH + 1)
    result = InputSanitizer.audit_text(long_text)
    assert result["safe"] is False
    assert "input_too_long" in result["issues"]
    
    # Test control chars
    result = InputSanitizer.audit_text("hello\x00world")
    assert result["safe"] is False
    assert "control_char" in result["issues"]
    
    # Test dangerous tokens
    dangerous_inputs = [
        "/bin/bash",
        "cmd.exe",
        "rm -rf /",
        "__import__",
        "eval('code')",
        "${command}",
        "`execute`"
    ]
    for text in dangerous_inputs:
        result = InputSanitizer.audit_text(text)
        assert result["safe"] is False
        assert any("danger_token:" in issue for issue in result["issues"])
        
def test_warnings():
    """Test detection of informational warnings."""
    # Test newlines
    result = InputSanitizer.audit_text("hello\nworld")
    assert result["safe"] is True  # Warnings don't affect safety
    assert "newline_present" in result["warnings"]
    
    # Test non-ASCII
    result = InputSanitizer.audit_text("café")
    assert result["safe"] is True
    assert "non_ascii_chars" in result["warnings"]
    
    # Test long lines
    long_line = "a" * 81
    result = InputSanitizer.audit_text(long_line)
    assert result["safe"] is True
    assert "long_lines" in result["warnings"]
    
    # Test repeated chars
    result = InputSanitizer.audit_text("hellooooo")
    assert result["safe"] is True
    assert "repeated_chars" in result["warnings"]
    
def test_multiple_issues():
    """Test handling of multiple issues and warnings."""
    text = f"/bin/bash\x00{'a' * 100}"
    result = InputSanitizer.audit_text(text)
    
    # Should detect multiple issues
    assert result["safe"] is False
    assert len(result["issues"]) >= 2
    
    # Verify specific issues
    issues = set(result["issues"])
    assert any("danger_token:" in issue for issue in issues)
    assert "control_char" in issues
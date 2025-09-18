"""
Input sanitization and validation for Aegis.

This module provides the InputSanitizer class which handles text
validation, normalization, and safety checks with a new warning
system that separates critical issues from informational warnings.
"""
import re
import unicodedata
from typing import Dict, Any, List, Set, Optional

from .sentinel_config import MAX_INPUT_LENGTH

class InputSanitizer:
    """Validates and sanitizes input text with warning system."""
    
    # Constants
    MAX_INPUT_LENGTH = MAX_INPUT_LENGTH
    
    # Control characters to block (except newline/tab)
    CONTROL_CHARS = set(chr(i) for i in range(32) if i not in (9, 10, 13))
    
    # Potentially dangerous tokens
    DANGEROUS_TOKENS = [
        r"(?<![\w/])/bin/(?!/)(?!.*\.)",  # Linux system binaries
        r"(?<![\w/])/sbin/(?!/)(?!.*\.)",  # Linux system admin binaries
        r"(?<![\w/])cmd\.exe(?![\w/])",    # Windows command shell
        r"(?<![\w/])powershell\.exe(?![\w/])",  # PowerShell
        r"(?<![\w/])bash\.exe(?![\w/])",   # WSL bash
        r"(?:[;|&]|\{\})",                 # Shell metacharacters
        r"[<>]",                           # Redirection
        r"\$[\(\{].*[\)\}]",              # Command substitution
        r"`.*`",                           # Backtick execution
        r"rm\s+-[rf]",                     # Dangerous rm flags
        r"chmod\s+777",                    # Overly permissive chmod
        r"eval\s*\(",                      # eval() calls
        r"exec\s*\(",                      # exec() calls
        r"system\s*\(",                    # system() calls
        r"__[a-zA-Z]+__"                  # Python magic methods
    ]
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for consistent processing.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text with consistent whitespace and encoding
            
        Raises:
            ValueError: If text cannot be normalized
        """
        try:
            # Convert to NFKC form
            text = unicodedata.normalize("NFKC", text)
            
            # Replace multiple spaces with single space
            text = re.sub(r"\s+", " ", text)
            
            # Normalize line endings
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            raise ValueError(f"Failed to normalize text: {str(e)}")
    
    @staticmethod
    def audit_text(text: str) -> Dict[str, Any]:
        """Audit input text for security and safety issues.
        
        This method separates issues into critical security problems
        that affect safety and informational warnings that don't.
        
        Args:
            text: Input text to audit
            
        Returns:
            Dict containing:
            - normalized: Normalized text
            - issues: List of critical issues
            - warnings: List of informational warnings
            - safe: Whether text is safe (based only on critical issues)
        """
        if not isinstance(text, str):
            return {
                "normalized": "",
                "issues": ["invalid_type"],
                "warnings": [],
                "safe": False
            }
            
        issues: Set[str] = set()
        warnings: Set[str] = set()
        
        # Check for critical issues
        if len(text) > InputSanitizer.MAX_INPUT_LENGTH:
            issues.add("input_too_long")
            
        for ch in InputSanitizer.CONTROL_CHARS:
            if ch in text:
                issues.add("control_char")
                break
                
        for tok in InputSanitizer.DANGEROUS_TOKENS:
            if re.search(tok, text, re.IGNORECASE):
                issues.add(f"danger_token:{tok}")
                
        # Check for informational warnings
        if "\n" in text or "\r" in text:
            warnings.add("newline_present")
            
        # Non-ASCII character warning
        if not text.isascii():
            warnings.add("non_ascii_chars")
            
        # Long line warning
        if any(len(line) > 80 for line in text.splitlines()):
            warnings.add("long_lines")
            
        # Repeated character warning
        if re.search(r"(.)\1{4,}", text):
            warnings.add("repeated_chars")
            
        try:
            normalized = InputSanitizer.normalize(text)
        except ValueError as e:
            issues.add(f"normalization_failed:{str(e)}")
            normalized = ""
            
        return {
            "normalized": normalized,
            "issues": sorted(issues),
            "warnings": sorted(warnings),
            "safe": len(issues) == 0  # Only critical issues affect safety
        }
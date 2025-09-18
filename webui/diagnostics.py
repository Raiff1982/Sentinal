"""
Diagnostics API endpoints for the Aegis system.

This module provides Flask endpoints for accessing diagnostic information
about the Aegis system, including resource usage, component health,
and security metrics.
"""

from flask import Blueprint, jsonify
from aegis.diagnostics import AegisDiagnostics

bp = Blueprint('diagnostics', __name__, url_prefix='/api/diagnostics')
diagnostics = AegisDiagnostics()

@bp.route('/', methods=['GET'])
def get_diagnostics():
    """Get all diagnostic information."""
    try:
        # Get resource metrics
        resources = diagnostics.get_resource_metrics()
        
        # Get component health
        components = diagnostics.check_critical_components()
        
        # Get active warnings
        warnings = diagnostics.get_active_warnings()
        
        # Get security metrics
        security = {
            "blocked_requests": diagnostics.security_metrics.blocked_requests,
            "warnings": len(diagnostics.security_metrics.recent_warnings),
            "allowed_requests": diagnostics.security_metrics.allowed_requests
        }
        
        # Get audit metrics from audit trail component health
        audit = components.get("audit_trail", {}).get("metrics", {})
        
        return jsonify({
            "resources": {
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "disk_usage": resources.disk_usage,
                "response_times": resources.response_times
            },
            "components": components,
            "warnings": warnings,
            "security": security,
            "audit": audit
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@bp.route('/resources', methods=['GET'])
def get_resources():
    """Get resource usage metrics."""
    try:
        metrics = diagnostics.get_resource_metrics()
        return jsonify({
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "disk_usage": metrics.disk_usage,
            "response_times": metrics.response_times
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/components', methods=['GET'])
def get_components():
    """Get component health status."""
    try:
        return jsonify(diagnostics.check_critical_components())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/warnings', methods=['GET'])
def get_warnings():
    """Get active system warnings."""
    try:
        return jsonify(diagnostics.get_active_warnings())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/security', methods=['GET'])
def get_security():
    """Get security metrics."""
    try:
        metrics = diagnostics.security_metrics
        return jsonify({
            "blocked_requests": metrics.blocked_requests,
            "recent_warnings": metrics.recent_warnings,
            "allowed_requests": metrics.allowed_requests,
            "risk_level": metrics.current_risk_level
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def init_app(app):
    """Initialize the diagnostics blueprint."""
    app.register_blueprint(bp)
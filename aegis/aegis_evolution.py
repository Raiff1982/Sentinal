"""
Evolutionary components for Aegis system adaptation.
"""
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class MetaGenes:
    """Represents evolvable parameters for meta-judgment."""
    
    # Weight coefficients for different signal types
    stress_weight: float = 0.25  # Weight for stress indicators
    risk_weight: float = 0.35    # Weight for risk indicators
    history_weight: float = 0.2  # Weight for historical patterns
    context_weight: float = 0.2  # Weight for contextual signals
    
    # Decision thresholds
    caution_threshold: float = 0.65  # When to trigger CAUTION
    block_threshold: float = 0.85    # When to trigger BLOCK
    
    # Influence modifiers
    short_term_influence: float = 0.4  # Short-term signal influence
    mid_term_influence: float = 0.35   # Mid-term signal influence
    long_term_influence: float = 0.25  # Long-term signal influence
    
    def evolve(self, performance_data: List[Dict[str, Any]]) -> 'MetaGenes':
        """Create evolved copy based on performance data.
        
        Args:
            performance_data: List of decision outcomes and their success metrics
            
        Returns:
            New MetaGenes instance with evolved parameters
        """
        # TODO: Implement evolution based on performance
        return MetaGenes(
            stress_weight=self.stress_weight,
            risk_weight=self.risk_weight,
            history_weight=self.history_weight,
            context_weight=self.context_weight,
            caution_threshold=self.caution_threshold,
            block_threshold=self.block_threshold,
            short_term_influence=self.short_term_influence,
            mid_term_influence=self.mid_term_influence,
            long_term_influence=self.long_term_influence
        )
        
    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return {
            "stress_weight": self.stress_weight,
            "risk_weight": self.risk_weight,
            "history_weight": self.history_weight,
            "context_weight": self.context_weight,
            "caution_threshold": self.caution_threshold,
            "block_threshold": self.block_threshold,
            "short_term_influence": self.short_term_influence,
            "mid_term_influence": self.mid_term_influence,
            "long_term_influence": self.long_term_influence
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'MetaGenes':
        """Create instance from dictionary."""
        return cls(**data)
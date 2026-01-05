# llm_driving/risk_calculator.py

"""
Risk-aware driving calculations integrated from the risk module.

This module provides a bridge between the risk module and the LLM driving pipeline,
enabling multi-dimensional risk scoring with:
- Time-to-Collision (TTC) calculations
- Object type weighting (pedestrians 2.5x)
- Multi-dimensional risk breakdown (collision, pedestrian, TTC, regulatory)
- Risk level classification (CRITICAL/HIGH/MODERATE/LOW/MINIMAL)
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict

import numpy as np

# Import from local module (now in same folder)
from .nuscenes_risk_integration import SimpleRiskCalculator, RiskComponents


# =============================================================================
# TYPE ID MAPPING (matches nuscenes_data.py)
# =============================================================================

TYPE_ID_TO_CATEGORY = {
    0: 'vehicle.car',
    1: 'human.pedestrian.adult',
    2: 'traffic_light',
    3: 'object.other',
}


# =============================================================================
# RISK CALCULATION FOR FRAME VECTORS
# =============================================================================

@dataclass
class FrameRiskData:
    """Risk data for an entire frame"""
    risk_level: str
    max_collision_risk: float
    max_pedestrian_risk: float
    min_ttc: Optional[float]
    avg_total_risk: float
    num_risk_objects: int
    object_risks: List[Dict]
    
    def to_dict(self) -> Dict:
        return asdict(self)


def calculate_risk_from_vectors(
    vectors: np.ndarray,
    num_objects: int,
    ego_speed: float = 10.0,
    traffic_light: Optional[str] = None,
) -> FrameRiskData:
    """
    Calculate comprehensive risk data from object vectors.
    
    Args:
        vectors: (MAX_OBJECTS, VECTOR_DIM) array with format:
                 [rel_x, rel_y, dist, rel_speed, heading, size, type_id]
        num_objects: Number of valid objects in vectors
        ego_speed: Current ego vehicle speed in m/s
        traffic_light: Optional traffic light state ('red', 'yellow', 'green')
    
    Returns:
        FrameRiskData containing multi-dimensional risk assessment
    """
    if num_objects == 0:
        return FrameRiskData(
            risk_level='MINIMAL',
            max_collision_risk=0.0,
            max_pedestrian_risk=0.0,
            min_ttc=None,
            avg_total_risk=0.0,
            num_risk_objects=0,
            object_risks=[],
        )
    
    calculator = SimpleRiskCalculator(ego_speed=ego_speed)
    
    object_risks = []
    total_risk_sum = 0.0
    max_collision = 0.0
    max_pedestrian = 0.0
    min_ttc = float('inf')
    
    use_n = min(int(num_objects), len(vectors))
    
    for i in range(use_n):
        vec = vectors[i]
        rel_x, rel_y, dist, rel_speed, heading, size, type_id = vec
        
        # Convert to risk calculator format
        position = np.array([float(rel_x), float(rel_y)])
        
        # Estimate velocity from relative speed and heading
        # Note: rel_speed is magnitude, heading is direction
        vx = float(rel_speed) * np.cos(float(heading))
        vy = float(rel_speed) * np.sin(float(heading))
        velocity = np.array([vx, vy])
        
        # Size as [width, length] - use size as approximation for both
        size_arr = np.array([float(size), float(size)])
        
        # Get object type from type_id
        obj_type = TYPE_ID_TO_CATEGORY.get(int(type_id), 'object.other')
        
        # Calculate risk
        risk_components, metadata = calculator.calculate_risk(
            position=position,
            velocity=velocity,
            size=size_arr,
            obj_type=obj_type,
            traffic_light=traffic_light,
        )
        
        # Aggregate statistics
        total_risk_sum += risk_components.total_risk
        max_collision = max(max_collision, risk_components.collision_risk)
        max_pedestrian = max(max_pedestrian, risk_components.pedestrian_risk)
        
        if metadata['ttc'] is not None and metadata['ttc'] < min_ttc:
            min_ttc = metadata['ttc']
        
        object_risks.append({
            'idx': i,
            'type': obj_type,
            'distance': metadata['distance'],
            'risk': risk_components.to_dict(),
            'ttc': metadata['ttc'],
            'closing_speed': metadata['closing_speed'],
        })
    
    # Calculate scene-level risk
    avg_risk = total_risk_sum / use_n if use_n > 0 else 0.0
    
    # Determine overall risk level
    if max_collision >= 0.8 or (min_ttc < float('inf') and min_ttc < 2):
        risk_level = 'CRITICAL'
    elif max_collision >= 0.6 or (min_ttc < float('inf') and min_ttc < 3):
        risk_level = 'HIGH'
    elif max_collision >= 0.4 or (min_ttc < float('inf') and min_ttc < 5):
        risk_level = 'MODERATE'
    elif max_collision >= 0.2:
        risk_level = 'LOW'
    else:
        risk_level = 'MINIMAL'
    
    return FrameRiskData(
        risk_level=risk_level,
        max_collision_risk=round(max_collision, 3),
        max_pedestrian_risk=round(max_pedestrian, 3),
        min_ttc=round(min_ttc, 2) if min_ttc < float('inf') else None,
        avg_total_risk=round(avg_risk, 3),
        num_risk_objects=use_n,
        object_risks=object_risks,
    )


# =============================================================================
# RISK-BASED POLICY DETERMINATION
# =============================================================================

def policy_from_risk(risk_data: FrameRiskData) -> Tuple[int, int, str, str, str]:
    """
    Determine driving policy based on multi-dimensional risk assessment.
    
    This replaces the simple distance-based _policy_from_min_dist().
    
    Args:
        risk_data: FrameRiskData from calculate_risk_from_vectors()
    
    Returns:
        Tuple of (accel, brake, steer, reason, policy_label)
    """
    steer = "straight"
    risk_level = risk_data.risk_level
    min_ttc = risk_data.min_ttc
    max_collision = risk_data.max_collision_risk
    max_pedestrian = risk_data.max_pedestrian_risk
    
    if risk_data.num_risk_objects == 0:
        return 20, 0, steer, "No nearby obstacles detected.", "CONTINUE"
    
    # CRITICAL: Emergency situation
    if risk_level == 'CRITICAL':
        if min_ttc is not None and min_ttc < 2:
            return 0, 90, steer, f"Critical risk: TTC={min_ttc:.1f}s, emergency braking required.", "BRAKE"
        return 0, 80, steer, f"Critical risk level (collision={max_collision:.0%}), braking hard.", "BRAKE"
    
    # HIGH: Significant danger
    if risk_level == 'HIGH':
        if max_pedestrian >= 0.5:
            return 0, 60, steer, f"High pedestrian risk ({max_pedestrian:.0%}), reducing speed.", "BRAKE"
        return 0, 50, steer, f"High risk (collision={max_collision:.0%}), slowing down.", "BRAKE"
    
    # MODERATE: Caution required
    if risk_level == 'MODERATE':
        if max_pedestrian >= 0.3:
            return 5, 30, steer, f"Moderate pedestrian risk, proceeding with caution.", "CAUTION"
        return 10, 20, steer, f"Moderate risk (collision={max_collision:.0%}), proceed carefully.", "CAUTION"
    
    # LOW: Minor concern
    if risk_level == 'LOW':
        return 15, 5, steer, f"Low risk detected, maintaining awareness.", "CONTINUE"
    
    # MINIMAL: Safe to proceed
    return 20, 0, steer, "Minimal risk, safe to continue.", "CONTINUE"


def get_risk_summary_text(risk_data: FrameRiskData) -> str:
    """
    Generate a human-readable risk summary for language generation.
    
    Args:
        risk_data: FrameRiskData from calculate_risk_from_vectors()
    
    Returns:
        String describing the overall risk situation
    """
    if risk_data.num_risk_objects == 0:
        return "The driving situation is clear with no significant risks."
    
    parts = [f"Risk level: {risk_data.risk_level}."]
    
    if risk_data.min_ttc is not None:
        parts.append(f"Minimum time-to-collision: {risk_data.min_ttc:.1f} seconds.")
    
    if risk_data.max_collision_risk >= 0.3:
        parts.append(f"Maximum collision risk: {risk_data.max_collision_risk:.0%}.")
    
    if risk_data.max_pedestrian_risk >= 0.3:
        parts.append(f"Pedestrian risk: {risk_data.max_pedestrian_risk:.0%}.")
    
    return " ".join(parts)

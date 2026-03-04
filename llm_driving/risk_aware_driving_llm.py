"""
Enhanced Risk-Aware Driving LLM System
======================================
This module implements a multi-dimensional risk scoring system
for autonomous driving with LLM integration.

Key Features:
1. Multi-dimensional risk vectors (not just single scalar)
2. Time-to-Collision (TTC) based risk
3. Object-type aware weighting
4. Risk attribution (which object contributes what %)
5. Temporal risk trajectory (how risk evolves)
6. Counterfactual risk reasoning
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

logger = logging.getLogger("llm_driving")

# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================

class RiskLevel(Enum):
    """Risk level categories for human-readable output"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MINIMAL = "MINIMAL"

@dataclass
class ObjectVector:
    """Structured representation of a detected object"""
    obj_id: int
    obj_type: str           # e.g., 'vehicle.car', 'human.pedestrian'
    position: np.ndarray    # [x, y] in meters from ego
    size: np.ndarray        # [width, length]
    velocity: np.ndarray    # [vx, vy] in m/s
    heading: float          # in radians

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.position)

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)

    @property
    def angle_degrees(self) -> float:
        return np.degrees(np.arctan2(self.position[1], self.position[0]))

@dataclass
class EgoState:
    """State of the ego vehicle"""
    velocity: np.ndarray    # [vx, vy] in m/s
    position: np.ndarray    # [x, y] - usually [0, 0]
    heading: float          # in radians

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)

@dataclass
class RiskVector:
    """Multi-dimensional risk representation"""
    collision_risk: float       # Risk of physical collision
    pedestrian_risk: float      # Risk involving vulnerable road users
    regulatory_risk: float      # Risk of violating traffic rules
    comfort_risk: float         # Risk of uncomfortable maneuvers needed
    uncertainty_risk: float     # Risk from perception uncertainty

    @property
    def total_risk(self) -> float:
        """Weighted combination of all risk dimensions"""
        weights = {
            'collision': 0.35,
            'pedestrian': 0.30,
            'regulatory': 0.15,
            'comfort': 0.10,
            'uncertainty': 0.10
        }
        return (
            self.collision_risk * weights['collision'] +
            self.pedestrian_risk * weights['pedestrian'] +
            self.regulatory_risk * weights['regulatory'] +
            self.comfort_risk * weights['comfort'] +
            self.uncertainty_risk * weights['uncertainty']
        )

    @property
    def risk_level(self) -> RiskLevel:
        """Convert total risk to categorical level"""
        total = self.total_risk
        if total >= 0.8:
            return RiskLevel.CRITICAL
        elif total >= 0.6:
            return RiskLevel.HIGH
        elif total >= 0.4:
            return RiskLevel.MODERATE
        elif total >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def to_dict(self) -> Dict:
        return {
            'collision_risk': round(self.collision_risk, 3),
            'pedestrian_risk': round(self.pedestrian_risk, 3),
            'regulatory_risk': round(self.regulatory_risk, 3),
            'comfort_risk': round(self.comfort_risk, 3),
            'uncertainty_risk': round(self.uncertainty_risk, 3),
            'total_risk': round(self.total_risk, 3),
            'risk_level': self.risk_level.value
        }

@dataclass
class ObjectRiskProfile:
    """Complete risk profile for a single object"""
    obj: ObjectVector
    risk_vector: RiskVector
    ttc: float                          # Time-to-collision in seconds
    risk_contribution: float            # % contribution to total scene risk
    recommended_action: str             # Suggested action for this object
    risk_trajectory: List[float]        # Predicted risk at t+1s, t+2s, t+3s

# =============================================================================
# PART 2: RISK CALCULATION ENGINE
# =============================================================================

class RiskCalculator:
    """
    Advanced risk calculation engine with multiple risk dimensions
    """

    # Object type weights (vulnerability/danger factor)
    TYPE_WEIGHTS = {
        'human.pedestrian': 2.5,        # Highest priority - vulnerable
        'human.pedestrian.child': 3.0,  # Children even more vulnerable
        'vehicle.bicycle': 2.0,
        'vehicle.motorcycle': 1.8,
        'vehicle.car': 1.0,
        'vehicle.truck': 1.3,
        'vehicle.bus': 1.3,
        'vehicle.emergency': 1.5,       # Emergency vehicles need attention
        'vehicle.construction': 1.2,
        'movable_object.trafficcone': 0.5,
        'static_object.bicycle_rack': 0.3,
    }

    SAFE_DISTANCE_FACTOR = 2.0  # seconds of following distance

    def __init__(self, ego_state: EgoState):
        self.ego = ego_state

    def get_type_weight(self, obj_type: str) -> float:
        """Get risk weight multiplier based on object type"""
        if obj_type in self.TYPE_WEIGHTS:
            return self.TYPE_WEIGHTS[obj_type]

        for key, weight in self.TYPE_WEIGHTS.items():
            if key in obj_type or obj_type in key:
                return weight

        return 1.0

    def calculate_ttc(self, obj: ObjectVector) -> float:
        """
        Calculate Time-to-Collision (TTC)

        TTC = distance / closing_speed
        Lower TTC = Higher risk
        """
        relative_velocity = obj.velocity - self.ego.velocity
        direction_to_obj = obj.position / (obj.distance + 1e-6)
        closing_speed = -np.dot(relative_velocity, direction_to_obj)

        if closing_speed <= 0:
            return float('inf')

        ttc = obj.distance / closing_speed
        return max(ttc, 0.01)

    def calculate_collision_risk(self, obj: ObjectVector, ttc: float) -> float:
        """Calculate collision risk based on TTC and distance"""
        if ttc == float('inf'):
            ttc_risk = 0.0
        elif ttc < 1.0:
            ttc_risk = 1.0
        elif ttc < 2.0:
            ttc_risk = 0.9
        elif ttc < 3.0:
            ttc_risk = 0.7
        elif ttc < 5.0:
            ttc_risk = 0.5
        else:
            ttc_risk = max(0, 1.0 - (ttc / 10.0))

        safe_distance = max(5.0, self.ego.speed * self.SAFE_DISTANCE_FACTOR)
        if obj.distance < safe_distance * 0.3:
            distance_risk = 1.0
        elif obj.distance < safe_distance * 0.6:
            distance_risk = 0.7
        elif obj.distance < safe_distance:
            distance_risk = 0.4
        else:
            distance_risk = max(0, 1.0 - (obj.distance / (safe_distance * 2)))

        type_weight = self.get_type_weight(obj.obj_type)
        raw_risk = max(ttc_risk, distance_risk)
        weighted_risk = min(1.0, raw_risk * (type_weight / 2.0 + 0.5))
        return weighted_risk

    def calculate_pedestrian_risk(self, obj: ObjectVector) -> float:
        """Special risk calculation for pedestrians and vulnerable road users"""
        if 'pedestrian' not in obj.obj_type.lower() and 'bicycle' not in obj.obj_type.lower():
            return 0.0

        if obj.distance > 15:
            return 0.1

        if obj.speed > 0.5:
            velocity_towards_ego = -np.dot(obj.velocity, obj.position) / (obj.distance + 1e-6)
            if velocity_towards_ego > 0:
                crossing_risk = min(1.0, velocity_towards_ego / 2.0)
            else:
                crossing_risk = 0.2
        else:
            crossing_risk = 0.3

        distance_factor = max(0, 1.0 - (obj.distance / 15.0))
        child_multiplier = 1.5 if 'child' in obj.obj_type.lower() else 1.0
        return min(1.0, (crossing_risk + distance_factor) * 0.5 * child_multiplier)

    def calculate_comfort_risk(self, obj: ObjectVector, ttc: float) -> float:
        """Risk of needing uncomfortable/harsh maneuvers"""
        if ttc == float('inf') or ttc > 5:
            return 0.0

        required_decel = (self.ego.speed ** 2) / (2 * obj.distance + 1e-6)

        if required_decel < 2.0:
            return 0.1
        elif required_decel < 4.0:
            return 0.3
        elif required_decel < 6.0:
            return 0.6
        elif required_decel < 8.0:
            return 0.8
        else:
            return 1.0

    def calculate_uncertainty_risk(self, obj: ObjectVector) -> float:
        """Risk from perception uncertainty"""
        distance_uncertainty = min(1.0, obj.distance / 50.0) * 0.5
        obj_area = obj.size[0] * obj.size[1]
        size_uncertainty = max(0, 0.5 - obj_area / 10.0)
        speed_uncertainty = min(0.3, obj.speed / 30.0)
        return min(1.0, distance_uncertainty + size_uncertainty + speed_uncertainty)

    def calculate_risk_trajectory(self, obj: ObjectVector, ttc: float) -> List[float]:
        """Predict how risk will evolve over next 3 seconds"""
        trajectory = []
        for dt in [1.0, 2.0, 3.0]:
            future_pos = obj.position + obj.velocity * dt
            ego_future_pos = self.ego.position + self.ego.velocity * dt
            future_distance = np.linalg.norm(future_pos - ego_future_pos)

            if future_distance < 2:
                future_risk = 1.0
            elif future_distance < 5:
                future_risk = 0.8
            elif future_distance < 10:
                future_risk = 0.5
            else:
                future_risk = max(0, 1.0 - future_distance / 30.0)

            trajectory.append(round(future_risk, 2))
        return trajectory

    def get_recommended_action(self, risk_vector: RiskVector, ttc: float) -> str:
        """Get recommended action based on risk profile"""
        total = risk_vector.total_risk
        if total >= 0.8 or ttc < 1.5:
            return "EMERGENCY_BRAKE"
        elif total >= 0.6 or ttc < 3.0:
            return "BRAKE_HARD"
        elif total >= 0.4 or ttc < 5.0:
            return "SLOW_DOWN"
        elif total >= 0.2:
            return "CAUTION"
        else:
            return "PROCEED"

    def calculate_object_risk(self, obj: ObjectVector,
                             traffic_light: Optional[str] = None) -> ObjectRiskProfile:
        """Calculate complete risk profile for a single object"""
        ttc = self.calculate_ttc(obj)

        risk_vector = RiskVector(
            collision_risk=self.calculate_collision_risk(obj, ttc),
            pedestrian_risk=self.calculate_pedestrian_risk(obj),
            regulatory_risk=0.0,
            comfort_risk=self.calculate_comfort_risk(obj, ttc),
            uncertainty_risk=self.calculate_uncertainty_risk(obj)
        )

        trajectory = self.calculate_risk_trajectory(obj, ttc)
        action = self.get_recommended_action(risk_vector, ttc)

        return ObjectRiskProfile(
            obj=obj,
            risk_vector=risk_vector,
            ttc=ttc if ttc != float('inf') else 999.9,
            risk_contribution=0.0,
            recommended_action=action,
            risk_trajectory=trajectory
        )

    def calculate_scene_risk(self, objects: List[ObjectVector],
                            traffic_light: Optional[str] = None,
                            speed_limit: Optional[float] = None) -> Dict:
        """Calculate comprehensive risk for entire scene"""
        object_risks = []
        for obj in objects:
            risk_profile = self.calculate_object_risk(obj, traffic_light)
            object_risks.append(risk_profile)

        if object_risks:
            total_collision = max(r.risk_vector.collision_risk for r in object_risks)
            total_pedestrian = max(r.risk_vector.pedestrian_risk for r in object_risks)
            total_comfort = max(r.risk_vector.comfort_risk for r in object_risks)
            total_uncertainty = np.mean([r.risk_vector.uncertainty_risk for r in object_risks])
        else:
            total_collision = 0.0
            total_pedestrian = 0.0
            total_comfort = 0.0
            total_uncertainty = 0.0

        regulatory_risk = 0.0
        if traffic_light == 'red':
            regulatory_risk = 0.9
        elif traffic_light == 'yellow':
            regulatory_risk = 0.5

        if speed_limit and self.ego.speed > speed_limit:
            speed_violation = (self.ego.speed - speed_limit) / speed_limit
            regulatory_risk = max(regulatory_risk, min(1.0, speed_violation))

        scene_risk_vector = RiskVector(
            collision_risk=total_collision,
            pedestrian_risk=total_pedestrian,
            regulatory_risk=regulatory_risk,
            comfort_risk=total_comfort,
            uncertainty_risk=total_uncertainty
        )

        total_raw_risk = sum(r.risk_vector.total_risk for r in object_risks) + 0.001
        for risk_profile in object_risks:
            risk_profile.risk_contribution = round(
                (risk_profile.risk_vector.total_risk / total_raw_risk) * 100, 1
            )

        object_risks.sort(key=lambda x: x.risk_contribution, reverse=True)

        top_factors = []
        if scene_risk_vector.collision_risk > 0.3:
            top_factors.append(f"Collision risk: {scene_risk_vector.collision_risk:.0%}")
        if scene_risk_vector.pedestrian_risk > 0.3:
            top_factors.append(f"Pedestrian risk: {scene_risk_vector.pedestrian_risk:.0%}")
        if scene_risk_vector.regulatory_risk > 0.3:
            top_factors.append(f"Regulatory risk: {scene_risk_vector.regulatory_risk:.0%}")
        if scene_risk_vector.comfort_risk > 0.3:
            top_factors.append(f"Comfort risk: {scene_risk_vector.comfort_risk:.0%}")

        min_ttc = min((r.ttc for r in object_risks), default=999)
        overall_action = self.get_recommended_action(scene_risk_vector, min_ttc)

        return {
            'scene_risk_vector': scene_risk_vector,
            'object_risks': object_risks,
            'top_risk_factors': top_factors[:3],
            'recommended_action': overall_action,
            'min_ttc': round(min_ttc, 1) if min_ttc < 999 else None
        }


# =============================================================================
# PART 3: LLM PROMPT GENERATION
# =============================================================================

class RiskAwarePromptGenerator:
    """Generates rich, risk-aware prompts for LLM"""

    def __init__(self):
        pass

    def format_risk_level_emoji(self, level: RiskLevel) -> str:
        mapping = {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MODERATE: "🟡",
            RiskLevel.LOW: "🟢",
            RiskLevel.MINIMAL: "⚪"
        }
        return mapping.get(level, "⚪")

    def format_trajectory(self, trajectory: List[float]) -> str:
        if len(trajectory) < 2:
            return "STABLE"

        trend = trajectory[-1] - trajectory[0]
        if trend > 0.2:
            return "↗️ ESCALATING"
        elif trend < -0.2:
            return "↘️ DECREASING"
        else:
            return "→ STABLE"

    def generate_basic_prompt(self, scene_risk: Dict, ego_speed: float) -> str:
        risk_vec = scene_risk['scene_risk_vector']

        prompt = f"""You are an autonomous driving AI assistant.

Current Speed: {ego_speed:.1f} m/s
Overall Risk Level: {risk_vec.risk_level.value} ({risk_vec.total_risk:.0%})

Scene Objects:
"""
        for obj_risk in scene_risk['object_risks'][:5]:
            obj = obj_risk.obj
            prompt += f"- {obj.obj_type}: {obj.distance:.1f}m away, risk={obj_risk.risk_vector.total_risk:.0%}\n"

        prompt += f"""
Recommended Action: {scene_risk['recommended_action']}

What action should the vehicle take and why?"""

        return prompt

    def generate_detailed_prompt(self, scene_risk: Dict, ego_speed: float,
                                 traffic_light: Optional[str] = None) -> str:
        risk_vec = scene_risk['scene_risk_vector']
        emoji = self.format_risk_level_emoji(risk_vec.risk_level)

        prompt = f"""You are an advanced autonomous driving AI with risk-aware reasoning capabilities.

═══════════════════════════════════════════════════════════════
                    CURRENT DRIVING SITUATION
═══════════════════════════════════════════════════════════════

EGO VEHICLE STATE:
  • Current Speed: {ego_speed:.1f} m/s ({ego_speed * 3.6:.1f} km/h)
  • Traffic Light: {traffic_light.upper() if traffic_light else 'N/A'}

───────────────────────────────────────────────────────────────
                    RISK ASSESSMENT SUMMARY
───────────────────────────────────────────────────────────────

{emoji} Overall Risk Level: {risk_vec.risk_level.value} ({risk_vec.total_risk:.0%})

Risk Breakdown:
  ├── Collision Risk:   {'█' * int(risk_vec.collision_risk * 10)}{'░' * (10 - int(risk_vec.collision_risk * 10))} {risk_vec.collision_risk:.0%}
  ├── Pedestrian Risk:  {'█' * int(risk_vec.pedestrian_risk * 10)}{'░' * (10 - int(risk_vec.pedestrian_risk * 10))} {risk_vec.pedestrian_risk:.0%}
  ├── Regulatory Risk:  {'█' * int(risk_vec.regulatory_risk * 10)}{'░' * (10 - int(risk_vec.regulatory_risk * 10))} {risk_vec.regulatory_risk:.0%}
  ├── Comfort Risk:     {'█' * int(risk_vec.comfort_risk * 10)}{'░' * (10 - int(risk_vec.comfort_risk * 10))} {risk_vec.comfort_risk:.0%}
  └── Uncertainty Risk: {'█' * int(risk_vec.uncertainty_risk * 10)}{'░' * (10 - int(risk_vec.uncertainty_risk * 10))} {risk_vec.uncertainty_risk:.0%}

"""
        if scene_risk['min_ttc']:
            prompt += f"⏱️  Minimum Time-to-Collision: {scene_risk['min_ttc']:.1f} seconds\n\n"

        if scene_risk['top_risk_factors']:
            prompt += "⚠️  Top Risk Factors:\n"
            for factor in scene_risk['top_risk_factors']:
                prompt += f"  • {factor}\n"
            prompt += "\n"

        prompt += """───────────────────────────────────────────────────────────────
                    DETECTED OBJECTS (by risk)
───────────────────────────────────────────────────────────────

"""
        for i, obj_risk in enumerate(scene_risk['object_risks'][:5], 1):
            obj = obj_risk.obj
            trajectory_str = self.format_trajectory(obj_risk.risk_trajectory)

            prompt += f"""Object {i}: {obj.obj_type.split('.')[-1].upper()}
  • Position: {obj.distance:.1f}m at {obj.angle_degrees:.0f}°
  • Speed: {obj.speed:.1f} m/s
  • TTC: {obj_risk.ttc:.1f}s
  • Risk: {obj_risk.risk_vector.total_risk:.0%} ({obj_risk.risk_contribution:.0f}% of total)
  • Trend: {trajectory_str}
  • Action: {obj_risk.recommended_action}

"""

        prompt += f"""───────────────────────────────────────────────────────────────
                    SYSTEM RECOMMENDATION
───────────────────────────────────────────────────────────────

Recommended Action: {scene_risk['recommended_action']}

═══════════════════════════════════════════════════════════════

Based on the above risk assessment, provide:
1. Your DECISION (one of: EMERGENCY_BRAKE, BRAKE_HARD, SLOW_DOWN, CAUTION, PROCEED, TURN_LEFT, TURN_RIGHT)
2. Your REASONING explaining why this is the safest action
3. Any ADDITIONAL PRECAUTIONS the vehicle should take

Response:"""

        return prompt

    def generate_counterfactual_prompt(self, scene_risk: Dict, ego_speed: float,
                                       risk_calculator: "RiskCalculator",
                                       objects: List[ObjectVector]) -> str:
        risk_vec = scene_risk['scene_risk_vector']

        actions_analysis = []
        actions_analysis.append({
            'action': 'CONTINUE',
            'risk_change': 0,
            'description': 'Maintain current speed'
        })

        braking_ego = EgoState(
            velocity=risk_calculator.ego.velocity * 0.5,
            position=risk_calculator.ego.position,
            heading=risk_calculator.ego.heading
        )
        brake_calc = RiskCalculator(braking_ego)
        brake_risk = brake_calc.calculate_scene_risk(objects)
        brake_change = brake_risk['scene_risk_vector'].total_risk - risk_vec.total_risk
        actions_analysis.append({
            'action': 'BRAKE',
            'new_risk': brake_risk['scene_risk_vector'].total_risk,
            'risk_change': brake_change,
            'description': 'Reduce speed by 50%'
        })

        hard_brake_ego = EgoState(
            velocity=risk_calculator.ego.velocity * 0.2,
            position=risk_calculator.ego.position,
            heading=risk_calculator.ego.heading
        )
        hard_brake_calc = RiskCalculator(hard_brake_ego)
        hard_brake_risk = hard_brake_calc.calculate_scene_risk(objects)
        hard_brake_change = hard_brake_risk['scene_risk_vector'].total_risk - risk_vec.total_risk
        actions_analysis.append({
            'action': 'HARD_BRAKE',
            'new_risk': hard_brake_risk['scene_risk_vector'].total_risk,
            'risk_change': hard_brake_change,
            'description': 'Reduce speed by 80%'
        })

        prompt = f"""You are an autonomous driving AI with predictive risk reasoning.

CURRENT SITUATION:
  • Speed: {ego_speed:.1f} m/s
  • Current Risk: {risk_vec.total_risk:.0%} ({risk_vec.risk_level.value})

COUNTERFACTUAL ANALYSIS - "What if I...?"
═══════════════════════════════════════════════════════════════

"""
        for analysis in actions_analysis:
            change = analysis['risk_change']
            if change < -0.1:
                change_str = f"↓ Risk decreases by {abs(change):.0%}"
            elif change > 0.1:
                change_str = f"↑ Risk increases by {abs(change):.0%}"
            else:
                change_str = "→ Risk unchanged"

            prompt += f"""If I {analysis['action']} ({analysis['description']}):
  {change_str}

"""

        prompt += f"""
DETECTED OBJECTS:
"""
        for obj_risk in scene_risk['object_risks'][:3]:
            obj = obj_risk.obj
            prompt += f"  • {obj.obj_type}: {obj.distance:.1f}m, risk={obj_risk.risk_vector.total_risk:.0%}\n"

        prompt += """
Based on this counterfactual analysis, what action should be taken and why?

Response:"""

        return prompt

    def generate_structured_output_prompt(self, scene_risk: Dict, ego_speed: float) -> str:
        risk_vec = scene_risk['scene_risk_vector']

        prompt = f"""You are an autonomous driving AI. Analyze the situation and provide a structured response.

INPUT:
{{
  "ego_speed_ms": {ego_speed:.1f},
  "risk_level": "{risk_vec.risk_level.value}",
  "total_risk": {risk_vec.total_risk:.2f},
  "collision_risk": {risk_vec.collision_risk:.2f},
  "pedestrian_risk": {risk_vec.pedestrian_risk:.2f},
  "regulatory_risk": {risk_vec.regulatory_risk:.2f},
  "min_ttc_seconds": {scene_risk['min_ttc'] if scene_risk['min_ttc'] else 'null'},
  "objects": [
"""
        for obj_risk in scene_risk['object_risks'][:5]:
            obj = obj_risk.obj
            prompt += f"""    {{
      "type": "{obj.obj_type}",
      "distance_m": {obj.distance:.1f},
      "speed_ms": {obj.speed:.1f},
      "risk": {obj_risk.risk_vector.total_risk:.2f},
      "ttc_s": {obj_risk.ttc:.1f}
    }},
"""

        prompt += """  ]
}

Respond with a JSON object containing:
{
  "action": "<EMERGENCY_BRAKE|BRAKE_HARD|SLOW_DOWN|CAUTION|PROCEED>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>",
  "accelerator_percent": <0-100>,
  "brake_percent": <0-100>,
  "steering": "<straight|left|right>"
}

Response:"""

        return prompt


# =============================================================================
# PART 4: INTEGRATION WITH NUSCENES
# =============================================================================

def extract_objects_from_nuscenes(nusc, sample) -> List[ObjectVector]:
    """Extract ObjectVector list from NuScenes sample"""
    objects = []

    for i, ann_token in enumerate(sample['anns']):
        ann = nusc.get('sample_annotation', ann_token)

        try:
            velocity = nusc.box_velocity(ann_token)[:2]
            if np.isnan(velocity).any():
                velocity = np.array([0.0, 0.0])
        except Exception:
            logger.exception("[risk_system] box_velocity failed for ann_token=%s", ann_token)
            velocity = np.array([0.0, 0.0])

        obj = ObjectVector(
            obj_id=i,
            obj_type=ann['category_name'],
            position=np.array(ann['translation'][:2]),
            size=np.array(ann['size'][:2]),
            velocity=velocity,
            heading=ann['rotation'][2] if len(ann['rotation']) > 2 else 0.0
        )
        objects.append(obj)

    return objects


def process_nuscenes_scene(nusc, scene_idx: int = 0,
                          ego_speed: float = 10.0) -> Tuple[Dict, str, str, str, str]:
    """
    Process a NuScenes scene and generate risk-aware prompts
    """
    logger.info("[risk_system] Processing scene_idx=%s ego_speed=%.2f", scene_idx, float(ego_speed))

    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    objects = extract_objects_from_nuscenes(nusc, sample)

    ego = EgoState(
        velocity=np.array([ego_speed, 0.0]),
        position=np.array([0.0, 0.0]),
        heading=0.0
    )

    calculator = RiskCalculator(ego)
    scene_risk = calculator.calculate_scene_risk(objects)

    prompt_gen = RiskAwarePromptGenerator()
    basic_prompt = prompt_gen.generate_basic_prompt(scene_risk, ego_speed)
    detailed_prompt = prompt_gen.generate_detailed_prompt(scene_risk, ego_speed)
    counterfactual_prompt = prompt_gen.generate_counterfactual_prompt(
        scene_risk, ego_speed, calculator, objects
    )
    structured_prompt = prompt_gen.generate_structured_output_prompt(scene_risk, ego_speed)

    logger.info(
        "[risk_system] scene_idx=%s recommended_action=%s risk_level=%s total_risk=%.3f min_ttc=%s",
        scene_idx,
        scene_risk.get("recommended_action"),
        scene_risk["scene_risk_vector"].risk_level.value,
        float(scene_risk["scene_risk_vector"].total_risk),
        scene_risk.get("min_ttc"),
    )

    return scene_risk, basic_prompt, detailed_prompt, counterfactual_prompt, structured_prompt


# =============================================================================
# PART 5: EXAMPLE USAGE
# =============================================================================

def demo_with_synthetic_data():
    """Demonstrate the system with synthetic data (no NuScenes needed)"""
    logger.info("=" * 70)
    logger.info("RISK-AWARE DRIVING LLM DEMO")
    logger.info("=" * 70)

    objects = [
        ObjectVector(
            obj_id=0,
            obj_type='human.pedestrian.adult',
            position=np.array([8.0, 2.0]),
            size=np.array([0.5, 0.5]),
            velocity=np.array([-1.0, 0.0]),
            heading=0.0
        ),
        ObjectVector(
            obj_id=1,
            obj_type='vehicle.car',
            position=np.array([15.0, 0.0]),
            size=np.array([2.0, 4.5]),
            velocity=np.array([-5.0, 0.0]),
            heading=np.pi
        ),
        ObjectVector(
            obj_id=2,
            obj_type='vehicle.bicycle',
            position=np.array([12.0, 3.0]),
            size=np.array([0.5, 1.8]),
            velocity=np.array([2.0, 0.0]),
            heading=0.0
        ),
    ]

    ego = EgoState(
        velocity=np.array([8.0, 0.0]),
        position=np.array([0.0, 0.0]),
        heading=0.0
    )

    calculator = RiskCalculator(ego)
    scene_risk = calculator.calculate_scene_risk(objects, traffic_light='green')

    prompt_gen = RiskAwarePromptGenerator()

    logger.info("\n" + "=" * 70)
    logger.info("1. BASIC PROMPT")
    logger.info("=" * 70)
    basic = prompt_gen.generate_basic_prompt(scene_risk, ego.speed)
    logger.info(basic)

    logger.info("\n" + "=" * 70)
    logger.info("2. DETAILED PROMPT")
    logger.info("=" * 70)
    detailed = prompt_gen.generate_detailed_prompt(scene_risk, ego.speed, 'green')
    logger.info(detailed)

    logger.info("\n" + "=" * 70)
    logger.info("3. COUNTERFACTUAL PROMPT")
    logger.info("=" * 70)
    counterfactual = prompt_gen.generate_counterfactual_prompt(
        scene_risk, ego.speed, calculator, objects
    )
    logger.info(counterfactual)

    logger.info("\n" + "=" * 70)
    logger.info("4. STRUCTURED OUTPUT PROMPT")
    logger.info("=" * 70)
    structured = prompt_gen.generate_structured_output_prompt(scene_risk, ego.speed)
    logger.info(structured)

    logger.info("\n" + "=" * 70)
    logger.info("RISK ANALYSIS SUMMARY")
    logger.info("=" * 70)
    rv = scene_risk['scene_risk_vector']
    logger.info("Total Risk: %.2f%%", rv.total_risk * 100.0)
    logger.info("Risk Level: %s", rv.risk_level.value)
    logger.info("Recommended Action: %s", scene_risk['recommended_action'])
    logger.info("Min TTC: %ss", str(scene_risk['min_ttc']))

    logger.info("Per-Object Risk Attribution:")
    for obj_risk in scene_risk['object_risks']:
        logger.info("  - %s: %.1f%%", obj_risk.obj.obj_type, float(obj_risk.risk_contribution))


if __name__ == "__main__":
    demo_with_synthetic_data()

"""
Integration: Risk-Aware Driving with NuScenes + Flan-T5
=======================================================

This file shows how to integrate the risk scoring system 
with your existing NuScenes pipeline and LLM training.

Drop-in replacement for your existing code!
"""

import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# =============================================================================
# SIMPLIFIED RISK CALCULATOR (Easy to integrate)
# =============================================================================

@dataclass
class RiskComponents:
    """Multi-dimensional risk breakdown"""
    collision_risk: float
    pedestrian_risk: float
    ttc_risk: float          # Time-to-collision based
    regulatory_risk: float
    total_risk: float
    risk_level: str
    
    def to_dict(self):
        return asdict(self)

class SimpleRiskCalculator:
    """
    Simplified risk calculator that works directly with NuScenes vectors
    
    This is designed to be a drop-in enhancement for your existing code.
    """
    
    # Risk weights for different object types
    TYPE_WEIGHTS = {
        'human.pedestrian': 2.5,
        'vehicle.bicycle': 2.0,
        'vehicle.motorcycle': 1.8,
        'vehicle.car': 1.0,
        'vehicle.truck': 1.3,
        'vehicle.bus': 1.3,
    }
    
    def __init__(self, ego_speed: float = 10.0):
        """
        Args:
            ego_speed: Current speed of ego vehicle in m/s
        """
        self.ego_speed = ego_speed
    
    def get_type_weight(self, obj_type: str) -> float:
        """Get risk multiplier based on object type"""
        for key, weight in self.TYPE_WEIGHTS.items():
            if key in obj_type.lower():
                return weight
        return 1.0
    
    def calculate_ttc(self, distance: float, relative_speed: float) -> float:
        """
        Calculate Time-to-Collision
        
        Args:
            distance: Distance to object in meters
            relative_speed: Closing speed (positive = approaching)
        
        Returns:
            TTC in seconds (float('inf') if not approaching)
        """
        if relative_speed <= 0:
            return float('inf')
        return distance / relative_speed
    
    def calculate_risk(self, 
                       position: np.ndarray,    # [x, y]
                       velocity: np.ndarray,    # [vx, vy]
                       size: np.ndarray,        # [width, length]
                       obj_type: str,
                       traffic_light: Optional[str] = None) -> Tuple[RiskComponents, Dict]:
        """
        Calculate comprehensive risk for a single object
        
        Args:
            position: [x, y] position relative to ego
            velocity: [vx, vy] velocity of object
            size: [width, length] of object
            obj_type: Category name (e.g., 'human.pedestrian.adult')
            traffic_light: Optional traffic light state ('red', 'yellow', 'green')
        
        Returns:
            RiskComponents: Multi-dimensional risk breakdown
            Dict: Additional risk metadata
        """
        # 1. Basic metrics
        distance = np.linalg.norm(position)
        speed = np.linalg.norm(velocity)
        
        # 2. Time-to-Collision
        # Relative velocity (closing speed)
        direction_to_obj = position / (distance + 1e-6)
        ego_velocity = np.array([self.ego_speed, 0])  # Assume driving forward
        relative_velocity = velocity - ego_velocity
        closing_speed = -np.dot(relative_velocity, direction_to_obj)
        
        ttc = self.calculate_ttc(distance, closing_speed)
        
        # 3. TTC-based risk
        if ttc == float('inf'):
            ttc_risk = 0.0
        elif ttc < 1.0:
            ttc_risk = 1.0
        elif ttc < 2.0:
            ttc_risk = 0.85
        elif ttc < 3.0:
            ttc_risk = 0.65
        elif ttc < 5.0:
            ttc_risk = 0.4
        else:
            ttc_risk = max(0, 0.3 - (ttc - 5) * 0.05)
        
        # 4. Distance-based collision risk
        safe_distance = max(5.0, self.ego_speed * 2.0)  # 2-second rule
        if distance < 3:
            distance_risk = 1.0
        elif distance < safe_distance * 0.5:
            distance_risk = 0.8
        elif distance < safe_distance:
            distance_risk = 0.5
        else:
            distance_risk = max(0, 0.3 * (1 - distance / 30))
        
        collision_risk = max(ttc_risk, distance_risk)
        
        # 5. Type-weighted collision risk
        type_weight = self.get_type_weight(obj_type)
        collision_risk = min(1.0, collision_risk * (0.5 + type_weight * 0.25))
        
        # 6. Pedestrian-specific risk
        pedestrian_risk = 0.0
        if 'pedestrian' in obj_type.lower() or 'bicycle' in obj_type.lower():
            if distance < 15:
                # Check if moving towards ego path
                lateral_velocity = abs(velocity[1]) if len(velocity) > 1 else 0
                if lateral_velocity > 0.5:  # Moving pedestrian
                    pedestrian_risk = min(1.0, (15 - distance) / 10 + lateral_velocity * 0.2)
                else:
                    pedestrian_risk = min(0.6, (15 - distance) / 15)
        
        # 7. Regulatory risk
        regulatory_risk = 0.0
        if traffic_light == 'red':
            regulatory_risk = 0.9
        elif traffic_light == 'yellow':
            regulatory_risk = 0.5
        
        # 8. Total risk (weighted combination)
        total_risk = (
            collision_risk * 0.40 +
            pedestrian_risk * 0.30 +
            ttc_risk * 0.20 +
            regulatory_risk * 0.10
        )
        
        # 9. Risk level classification
        if total_risk >= 0.8:
            risk_level = "CRITICAL"
        elif total_risk >= 0.6:
            risk_level = "HIGH"
        elif total_risk >= 0.4:
            risk_level = "MODERATE"
        elif total_risk >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        risk_components = RiskComponents(
            collision_risk=round(collision_risk, 3),
            pedestrian_risk=round(pedestrian_risk, 3),
            ttc_risk=round(ttc_risk, 3),
            regulatory_risk=round(regulatory_risk, 3),
            total_risk=round(total_risk, 3),
            risk_level=risk_level
        )
        
        metadata = {
            'distance': round(distance, 2),
            'speed': round(speed, 2),
            'ttc': round(ttc, 2) if ttc < 100 else None,
            'type_weight': type_weight,
            'closing_speed': round(closing_speed, 2)
        }
        
        return risk_components, metadata


# =============================================================================
# NUSCENES INTEGRATION
# =============================================================================

def extract_scene_with_advanced_risk(nusc, scene_idx: int, 
                                      ego_speed: float = 10.0,
                                      traffic_light: Optional[str] = None) -> Dict:
    """
    Extract scene from NuScenes with advanced risk scoring
    
    This replaces your existing extract_scene_with_risk function
    
    Args:
        nusc: NuScenes instance
        scene_idx: Index of scene to process
        ego_speed: Assumed ego vehicle speed (m/s)
        traffic_light: Optional traffic light state
    
    Returns:
        Dict containing scene data with rich risk information
    """
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    
    if not sample['anns']:
        return None
    
    calculator = SimpleRiskCalculator(ego_speed=ego_speed)
    
    objects_with_risk = []
    total_scene_risk = 0
    max_collision_risk = 0
    max_pedestrian_risk = 0
    min_ttc = float('inf')
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # Extract object data
        position = np.array(ann['translation'][:2])
        size = np.array(ann['size'][:2])
        
        # Get velocity (handle missing data)
        try:
            velocity = nusc.box_velocity(ann_token)[:2]
            if np.isnan(velocity).any():
                velocity = np.array([0.0, 0.0])
        except:
            velocity = np.array([0.0, 0.0])
        
        obj_type = ann['category_name']
        
        # Calculate risk
        risk_components, metadata = calculator.calculate_risk(
            position=position,
            velocity=velocity,
            size=size,
            obj_type=obj_type,
            traffic_light=traffic_light
        )
        
        objects_with_risk.append({
            'type': obj_type,
            'position': position.tolist(),
            'velocity': velocity.tolist(),
            'size': size.tolist(),
            'risk': risk_components.to_dict(),
            'metadata': metadata
        })
        
        # Track scene-level stats
        total_scene_risk += risk_components.total_risk
        max_collision_risk = max(max_collision_risk, risk_components.collision_risk)
        max_pedestrian_risk = max(max_pedestrian_risk, risk_components.pedestrian_risk)
        if metadata['ttc'] is not None:
            min_ttc = min(min_ttc, metadata['ttc'])
    
    # Sort by risk
    objects_with_risk.sort(key=lambda x: x['risk']['total_risk'], reverse=True)
    
    # Calculate risk attribution (% contribution per object)
    if total_scene_risk > 0:
        for obj in objects_with_risk:
            obj['risk_attribution'] = round(
                (obj['risk']['total_risk'] / total_scene_risk) * 100, 1
            )
    
    # Scene-level risk assessment
    num_objects = len(objects_with_risk)
    avg_risk = total_scene_risk / num_objects if num_objects > 0 else 0
    
    scene_risk_level = "MINIMAL"
    if max_collision_risk >= 0.8 or min_ttc < 2:
        scene_risk_level = "CRITICAL"
    elif max_collision_risk >= 0.6 or min_ttc < 3:
        scene_risk_level = "HIGH"
    elif max_collision_risk >= 0.4 or min_ttc < 5:
        scene_risk_level = "MODERATE"
    elif max_collision_risk >= 0.2:
        scene_risk_level = "LOW"
    
    return {
        'scene_idx': scene_idx,
        'ego_speed': ego_speed,
        'traffic_light': traffic_light,
        'objects': objects_with_risk,
        'scene_summary': {
            'num_objects': num_objects,
            'risk_level': scene_risk_level,
            'max_collision_risk': round(max_collision_risk, 3),
            'max_pedestrian_risk': round(max_pedestrian_risk, 3),
            'min_ttc': round(min_ttc, 2) if min_ttc < 100 else None,
            'avg_risk': round(avg_risk, 3)
        }
    }


# =============================================================================
# PROMPT GENERATION FOR LLM
# =============================================================================

def generate_risk_aware_prompt(scene_data: Dict, prompt_style: str = 'detailed') -> str:
    """
    Generate LLM prompt with risk information
    
    Args:
        scene_data: Output from extract_scene_with_advanced_risk
        prompt_style: One of 'basic', 'detailed', 'structured'
    
    Returns:
        Formatted prompt string for LLM
    """
    if prompt_style == 'basic':
        return _generate_basic_prompt(scene_data)
    elif prompt_style == 'detailed':
        return _generate_detailed_prompt(scene_data)
    elif prompt_style == 'structured':
        return _generate_structured_prompt(scene_data)
    else:
        return _generate_detailed_prompt(scene_data)


def _generate_basic_prompt(scene_data: Dict) -> str:
    """Basic prompt - minimal risk info"""
    summary = scene_data['scene_summary']
    
    prompt = f"""You are an autonomous driving assistant.

Ego Speed: {scene_data['ego_speed']:.1f} m/s
Overall Risk: {summary['risk_level']} (collision={summary['max_collision_risk']:.0%})

Objects detected:
"""
    for obj in scene_data['objects'][:5]:
        prompt += f"- {obj['type']}: {obj['metadata']['distance']:.1f}m, risk={obj['risk']['total_risk']:.0%}\n"
    
    prompt += "\nDecide: STOP, SLOW_DOWN, or PROCEED. Explain your reasoning."
    
    return prompt


def _generate_detailed_prompt(scene_data: Dict) -> str:
    """Detailed prompt with full risk breakdown"""
    summary = scene_data['scene_summary']
    
    # Risk level emoji
    emoji_map = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MODERATE': '🟡', 'LOW': '🟢', 'MINIMAL': '⚪'}
    emoji = emoji_map.get(summary['risk_level'], '⚪')
    
    prompt = f"""You are an advanced autonomous driving AI with risk-aware reasoning.

══════════════════════════════════════════════════════════════
                     DRIVING SITUATION ANALYSIS
══════════════════════════════════════════════════════════════

EGO VEHICLE:
  • Speed: {scene_data['ego_speed']:.1f} m/s ({scene_data['ego_speed'] * 3.6:.1f} km/h)
  • Traffic Light: {scene_data['traffic_light'] or 'N/A'}

RISK ASSESSMENT:
  {emoji} Overall: {summary['risk_level']}
  • Max Collision Risk: {summary['max_collision_risk']:.0%}
  • Max Pedestrian Risk: {summary['max_pedestrian_risk']:.0%}
  • Min Time-to-Collision: {summary['min_ttc']}s

DETECTED OBJECTS (sorted by risk):
"""
    
    for i, obj in enumerate(scene_data['objects'][:5], 1):
        risk = obj['risk']
        meta = obj['metadata']
        
        prompt += f"""
  [{i}] {obj['type'].split('.')[-1].upper()}
      Distance: {meta['distance']:.1f}m | Speed: {meta['speed']:.1f}m/s
      TTC: {meta['ttc'] or 'N/A'}s | Closing: {meta['closing_speed']:.1f}m/s
      Risk: {risk['total_risk']:.0%} ({risk['risk_level']})
      └─ Collision: {risk['collision_risk']:.0%} | Pedestrian: {risk['pedestrian_risk']:.0%}
      Contribution: {obj.get('risk_attribution', 0):.0f}% of total scene risk
"""
    
    prompt += f"""
══════════════════════════════════════════════════════════════

Based on this risk assessment:
1. What ACTION should the vehicle take? (EMERGENCY_BRAKE / BRAKE / SLOW_DOWN / PROCEED)
2. Provide REASONING for your decision
3. What PRECAUTIONS should be taken?

Response:"""
    
    return prompt


def _generate_structured_prompt(scene_data: Dict) -> str:
    """Structured prompt requesting JSON output"""
    summary = scene_data['scene_summary']
    
    prompt = f"""Analyze this driving scene and respond with a JSON decision.

INPUT:
{{
  "ego_speed_ms": {scene_data['ego_speed']},
  "scene_risk_level": "{summary['risk_level']}",
  "max_collision_risk": {summary['max_collision_risk']},
  "max_pedestrian_risk": {summary['max_pedestrian_risk']},
  "min_ttc_seconds": {summary['min_ttc'] or 'null'},
  "objects": [
"""
    
    for obj in scene_data['objects'][:5]:
        prompt += f"""    {{
      "type": "{obj['type']}",
      "distance_m": {obj['metadata']['distance']},
      "risk": {obj['risk']['total_risk']},
      "ttc_s": {obj['metadata']['ttc'] or 'null'}
    }},
"""
    
    prompt += """  ]
}

OUTPUT FORMAT:
{
  "action": "EMERGENCY_BRAKE|BRAKE|SLOW_DOWN|PROCEED",
  "brake_intensity": 0-100,
  "steering": "straight|left|right",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Response:"""
    
    return prompt


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_sample(scene_data: Dict, 
                              ground_truth_action: Optional[str] = None) -> Dict:
    """
    Generate a training sample for LLM fine-tuning
    
    Args:
        scene_data: Scene with risk data
        ground_truth_action: Optional ground truth action label
    
    Returns:
        Dict with input prompt and target output
    """
    prompt = generate_risk_aware_prompt(scene_data, prompt_style='detailed')
    
    # If no ground truth, generate based on risk
    if ground_truth_action is None:
        summary = scene_data['scene_summary']
        if summary['risk_level'] == 'CRITICAL' or (summary['min_ttc'] and summary['min_ttc'] < 2):
            ground_truth_action = "EMERGENCY_BRAKE"
            reasoning = "Critical risk detected with very low time-to-collision."
        elif summary['risk_level'] == 'HIGH' or (summary['min_ttc'] and summary['min_ttc'] < 4):
            ground_truth_action = "BRAKE"
            reasoning = "High risk objects detected requiring immediate speed reduction."
        elif summary['risk_level'] == 'MODERATE':
            ground_truth_action = "SLOW_DOWN"
            reasoning = "Moderate risk present, reducing speed as precaution."
        else:
            ground_truth_action = "PROCEED"
            reasoning = "Low risk situation, safe to continue at current speed."
    else:
        reasoning = "Action determined by ground truth."
    
    target = f"""Action: {ground_truth_action}

Reasoning: {reasoning}

The decision is based on:
- Scene risk level: {scene_data['scene_summary']['risk_level']}
- Maximum collision risk: {scene_data['scene_summary']['max_collision_risk']:.0%}
- Minimum time-to-collision: {scene_data['scene_summary']['min_ttc'] or 'N/A'}s
- Number of risk objects: {scene_data['scene_summary']['num_objects']}"""
    
    return {
        'input': prompt,
        'output': target,
        'metadata': {
            'scene_idx': scene_data['scene_idx'],
            'risk_level': scene_data['scene_summary']['risk_level'],
            'action': ground_truth_action
        }
    }


def build_risk_aware_dataset(nusc, num_scenes: int = None, 
                              ego_speed: float = 10.0) -> List[Dict]:
    """
    Build complete training dataset with risk-aware prompts
    
    Args:
        nusc: NuScenes instance
        num_scenes: Number of scenes to process (None = all)
        ego_speed: Assumed ego speed
    
    Returns:
        List of training samples
    """
    if num_scenes is None:
        num_scenes = len(nusc.scene)
    
    dataset = []
    
    for scene_idx in range(min(num_scenes, len(nusc.scene))):
        scene_data = extract_scene_with_advanced_risk(
            nusc, scene_idx, ego_speed=ego_speed
        )
        
        if scene_data is None:
            continue
        
        sample = generate_training_sample(scene_data)
        dataset.append(sample)
    
    return dataset


# =============================================================================
# EXAMPLE: COMPLETE INTEGRATION WITH YOUR EXISTING CODE
# =============================================================================

def example_integration():
    """
    Example showing how to integrate with your existing NuScenes + Flan-T5 setup
    """
    print("=" * 70)
    print("EXAMPLE: Risk-Aware Driving LLM Integration")
    print("=" * 70)
    
    # Simulated scene data (replace with real NuScenes data)
    scene_data = {
        'scene_idx': 0,
        'ego_speed': 8.5,
        'traffic_light': 'green',
        'objects': [
            {
                'type': 'human.pedestrian.adult',
                'position': [10.0, 3.0],
                'velocity': [-0.5, -1.2],
                'size': [0.5, 0.5],
                'risk': {
                    'collision_risk': 0.65,
                    'pedestrian_risk': 0.72,
                    'ttc_risk': 0.55,
                    'regulatory_risk': 0.0,
                    'total_risk': 0.58,
                    'risk_level': 'MODERATE'
                },
                'metadata': {
                    'distance': 10.4,
                    'speed': 1.3,
                    'ttc': 4.2,
                    'type_weight': 2.5,
                    'closing_speed': 2.1
                },
                'risk_attribution': 45.0
            },
            {
                'type': 'vehicle.car',
                'position': [18.0, 0.0],
                'velocity': [-3.0, 0.0],
                'size': [2.0, 4.5],
                'risk': {
                    'collision_risk': 0.42,
                    'pedestrian_risk': 0.0,
                    'ttc_risk': 0.35,
                    'regulatory_risk': 0.0,
                    'total_risk': 0.32,
                    'risk_level': 'LOW'
                },
                'metadata': {
                    'distance': 18.0,
                    'speed': 3.0,
                    'ttc': 6.5,
                    'type_weight': 1.0,
                    'closing_speed': 1.5
                },
                'risk_attribution': 32.0
            },
            {
                'type': 'vehicle.bicycle',
                'position': [12.0, 4.0],
                'velocity': [1.5, 0.0],
                'size': [0.5, 1.8],
                'risk': {
                    'collision_risk': 0.38,
                    'pedestrian_risk': 0.45,
                    'ttc_risk': 0.28,
                    'regulatory_risk': 0.0,
                    'total_risk': 0.35,
                    'risk_level': 'LOW'
                },
                'metadata': {
                    'distance': 12.6,
                    'speed': 1.5,
                    'ttc': 8.2,
                    'type_weight': 2.0,
                    'closing_speed': 1.0
                },
                'risk_attribution': 23.0
            }
        ],
        'scene_summary': {
            'num_objects': 3,
            'risk_level': 'MODERATE',
            'max_collision_risk': 0.65,
            'max_pedestrian_risk': 0.72,
            'min_ttc': 4.2,
            'avg_risk': 0.42
        }
    }
    
    # Generate different prompt styles
    print("\n" + "=" * 70)
    print("1. BASIC PROMPT")
    print("=" * 70)
    basic_prompt = generate_risk_aware_prompt(scene_data, 'basic')
    print(basic_prompt)
    
    print("\n" + "=" * 70)
    print("2. DETAILED PROMPT")
    print("=" * 70)
    detailed_prompt = generate_risk_aware_prompt(scene_data, 'detailed')
    print(detailed_prompt)
    
    print("\n" + "=" * 70)
    print("3. STRUCTURED PROMPT")
    print("=" * 70)
    structured_prompt = generate_risk_aware_prompt(scene_data, 'structured')
    print(structured_prompt)
    
    print("\n" + "=" * 70)
    print("4. TRAINING SAMPLE")
    print("=" * 70)
    training_sample = generate_training_sample(scene_data)
    print("Input length:", len(training_sample['input']))
    print("Output:\n", training_sample['output'])
    print("Metadata:", training_sample['metadata'])


if __name__ == "__main__":
    example_integration()

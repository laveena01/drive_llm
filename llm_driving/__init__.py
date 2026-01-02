# llm_driving/__init__.py

"""
Package for a small nuScenes-mini + LLM driving prototype.

Modules:
- config: global constants and paths
- nuscenes_data: object-level vector extraction from nuScenes
- langen: vector -> language (lanGen-style) functions
- datasets_builder: builds captioning + driving QA datasets
- training: training loops for Stage 1 (captioning) & Stage 2 (QA)
"""

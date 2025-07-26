import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_imports():
    import main
    import preprocessing.face_detection
    import preprocessing.lip_extraction
    import preprocessing.video_utils
    import feature_extraction.visual_features
    import feature_extraction.temporal_features
    import synthesis.speech_synthesis
    import synthesis.vocoder
    import training.train
    import training.evaluate
    import utils.config
    import utils.helpers
    import utils.logger

def test_pipeline_init():
    from main import LipSyncAI
    lipsync = LipSyncAI()
    assert lipsync is not None 
import numpy as np
from collections import defaultdict
from typing import Dict

class TemporalPatternDetector:
    """
    Detects coordinated posting patterns over time
    """
    def __init__(self, time_window: int = 3600):
        self.time_window = time_window  # in seconds
        self.post_records = defaultdict(list)

    def add_post(self, content_hash: str, timestamp: float):
        self.post_records[content_hash].append(timestamp)

    def detect_coordination(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for content, timestamps in self.post_records.items():
            if len(timestamps) < 3:
                continue
            intervals = np.diff(sorted(timestamps))
            if np.std(intervals) < self.time_window / 3:
                results[content] = {
                    'post_count': len(timestamps),
                    'interval_std': float(np.std(intervals)),
                    'time_range': float(max(timestamps) - min(timestamps))
                }
        return results

class LinguisticStyleAnalyzer:
    """
    Linguistic fingerprint analysis for fake news detection
    """
    def __init__(self):
        self.style_markers = {
            'sensationalism': ['!', 'ALL CAPS', 'emotive adjectives'],
            'vagueness': ['many', 'some', 'experts say'],
            'false_urgency': ['now', 'urgent', 'breaking']
        }

    def analyze_style(self, text: str) -> Dict[str, float]:
        scores = {}
        text_lower = text.lower()
        for category, markers in self.style_markers.items():
            scores[category] = sum(
                1 for marker in markers if marker.lower() in text_lower
            ) / len(markers)
        return scores

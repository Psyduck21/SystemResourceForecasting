#!/usr/bin/env python3
"""
Script to test and fix anomaly detection issues
"""

import sys
import time
from ml_system_manager import MLSystemManager

def test_threshold_fixes():
    """Test different threshold values"""
    print("🧪 TESTING ANOMALY THRESHOLDS")

    ml_manager = MLSystemManager()
    ml_manager.start_monitoring()
    ml_manager.start_anomaly_detection()

    # Test with lower threshold (0.3 instead of 0.6)
    original_threshold = ml_manager.config["anomaly_score_threshold"]
    print(f"📊 Original threshold: {original_threshold}")

    # Temporarily lower threshold
    ml_manager.config["anomaly_score_threshold"] = 0.3
    print("📊 Testing with lower threshold: 0.3")

    # Test anomaly detection
    print("\n🧪 Testing with lowered threshold...")
    anomalies = ml_manager.check_process_anomalies()

    if anomalies:
        print(f"✅ Found {len(anomalies)} anomalies with lower threshold!")
        for anomaly in anomalies:
            print(f"   🚨 {anomaly.process_name}: score={anomaly.anomaly_score:.2f}, type={anomaly.anomaly_type}")
    else:
        print("❌ Still no anomalies even with lower threshold")

    # Test with very low threshold (0.1)
    ml_manager.config["anomaly_score_threshold"] = 0.1
    print("\n📊 Testing with very low threshold: 0.1")

    anomalies = ml_manager.check_process_anomalies()

    if anomalies:
        print(f"✅ Found {len(anomalies)} anomalies with very low threshold!")
        for anomaly in anomalies:
            print(f"   🚨 {anomaly.process_name}: score={anomaly.anomaly_score:.2f}, type={anomaly.anomaly_type}")
    else:
        print("❌ Still no anomalies - there may be a deeper issue")

    # Restore original threshold
    ml_manager.config["anomaly_score_threshold"] = original_threshold
    print(f"\n📊 Restored original threshold: {original_threshold}")

    ml_manager.cleanup()

if __name__ == "__main__":
    test_threshold_fixes()

"""
Advanced Model Monitoring and Tracking System
==============================================

This module implements comprehensive model monitoring using:
- MLflow for experiment tracking and model registry
- Prometheus metrics for real-time monitoring
- Data drift detection using statistical methods
- Automated retraining alerts
- Model performance degradation detection

Features:
- Real-time prediction accuracy monitoring
- Data drift detection with KS-test and PSI
- Model performance alerts
- Automated model retraining triggers
- Comprehensive logging and reporting
"""

import os
import logging
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import mlflow
import mlflow.sklearn
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, push_to_gateway
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Comprehensive model monitoring system
    """
    
    def __init__(self, model_name: str = "air_quality_lstm", 
                 tracking_uri: str = "sqlite:///mlflow_tracking.db",
                 prometheus_gateway: Optional[str] = None):
        """
        Initialize model monitoring system
        
        Args:
            model_name: Name of the model to monitor
            tracking_uri: MLflow tracking server URI
            prometheus_gateway: Prometheus pushgateway URL (optional)
        """
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.prometheus_gateway = prometheus_gateway
        
        # Initialize MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"{model_name}_monitoring")
        
        # Create Prometheus registry
        self.registry = CollectorRegistry()
        
        # Prometheus metrics
        self.prediction_counter = Counter(
            'model_predictions_total', 
            'Total predictions made',
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_duration_seconds',
            'Model prediction latency',
            registry=self.registry
        )
        
        self.model_accuracy_gauge = Gauge(
            'model_accuracy_score',
            'Current model accuracy (R²)',
            registry=self.registry
        )
        
        self.data_drift_score = Gauge(
            'model_data_drift_score',
            'Data drift detection score',
            registry=self.registry
        )
        
        self.model_mae = Gauge(
            'model_mae',
            'Model Mean Absolute Error',
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'model_errors_total',
            'Total model errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Data storage for drift detection
        self.reference_data = None
        self.recent_predictions = []
        self.recent_actuals = []
        self.recent_features = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'accuracy_drop': 0.05,      # Alert if R² drops by 5%
            'drift_threshold': 0.1,     # PSI threshold for drift
            'error_rate_threshold': 0.1, # 10% error rate threshold
            'latency_threshold': 200,   # 200ms latency threshold
            'sample_size_min': 100      # Minimum samples for drift detection
        }
        
        # Performance baselines
        self.baseline_metrics = {
            'r2_score': 0.90,
            'mae': 5.0,
            'rmse': 7.0,
            'latency_ms': 100.0
        }
        
        logger.info(f"ModelMonitor initialized for {model_name}")
    
    def log_prediction(self, features: np.ndarray, prediction: float, 
                      actual: Optional[float] = None, latency_ms: float = 0.0,
                      model_version: str = "v1.0") -> Dict[str, Any]:
        """
        Log a prediction with monitoring data
        
        Args:
            features: Input features used for prediction
            prediction: Model prediction value
            actual: Actual/ground truth value (if available)
            latency_ms: Prediction latency in milliseconds
            model_version: Version of model used
            
        Returns:
            Dictionary with monitoring results
        """
        timestamp = datetime.now()
        
        # Update Prometheus metrics
        self.prediction_counter.inc()
        self.prediction_latency.observe(latency_ms / 1000.0)
        
        # Store data for analysis
        self.recent_features.append(features.tolist())
        self.recent_predictions.append(prediction)
        
        if actual is not None:
            self.recent_actuals.append(actual)
            
            # Calculate current accuracy if we have enough samples
            if len(self.recent_actuals) >= 10:
                recent_r2 = r2_score(
                    self.recent_actuals[-10:], 
                    self.recent_predictions[-10:]
                )
                recent_mae = mean_absolute_error(
                    self.recent_actuals[-10:], 
                    self.recent_predictions[-10:]
                )
                
                # Update metrics
                self.model_accuracy_gauge.set(recent_r2)
                self.model_mae.set(recent_mae)
        
        # Keep only recent data (last 1000 predictions)
        max_history = 1000
        if len(self.recent_predictions) > max_history:
            self.recent_features = self.recent_features[-max_history:]
            self.recent_predictions = self.recent_predictions[-max_history:]
            if self.recent_actuals:
                self.recent_actuals = self.recent_actuals[-max_history:]
        
        # Log to MLflow
        prediction_data = {
            'timestamp': timestamp.isoformat(),
            'prediction': prediction,
            'actual': actual,
            'latency_ms': latency_ms,
            'model_version': model_version,
            'feature_stats': {
                'mean': float(np.mean(features)),
                'std': float(np.std(features)),
                'min': float(np.min(features)),
                'max': float(np.max(features))
            }
        }
        
        # Periodic MLflow logging (every 100 predictions)
        if len(self.recent_predictions) % 100 == 0:
            self._log_batch_metrics()
        
        return prediction_data
    
    def detect_data_drift(self, current_features: np.ndarray,
                         reference_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect data drift using statistical methods
        
        Args:
            current_features: Recent feature data
            reference_features: Reference/baseline feature data
            
        Returns:
            Dictionary with drift detection results
        """
        if reference_features is None:
            if self.reference_data is None:
                logger.warning("No reference data available for drift detection")
                return {'drift_detected': False, 'reason': 'no_reference_data'}
            reference_features = self.reference_data
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drifts': {},
            'method': 'kolmogorov_smirnov_test'
        }
        
        # Ensure we have enough samples
        if len(current_features) < self.alert_thresholds['sample_size_min']:
            drift_results['reason'] = 'insufficient_samples'
            return drift_results
        
        # Perform drift detection for each feature
        n_features = min(current_features.shape[1], reference_features.shape[1])
        drift_scores = []
        
        for i in range(n_features):
            current_feature = current_features[:, i]
            reference_feature = reference_features[:, i]
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(reference_feature, current_feature)
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(reference_feature, current_feature)
            
            feature_drift = {
                'ks_statistic': float(ks_statistic),
                'ks_p_value': float(p_value),
                'psi_score': float(psi_score),
                'drift_detected': psi_score > self.alert_thresholds['drift_threshold']
            }
            
            drift_results['feature_drifts'][f'feature_{i}'] = feature_drift
            drift_scores.append(psi_score)
        
        # Overall drift score
        overall_drift_score = np.mean(drift_scores)
        drift_results['drift_score'] = float(overall_drift_score)
        drift_results['drift_detected'] = overall_drift_score > self.alert_thresholds['drift_threshold']
        
        # Update Prometheus metric
        self.data_drift_score.set(overall_drift_score)
        
        # Log to MLflow if drift detected
        if drift_results['drift_detected']:
            with mlflow.start_run():
                mlflow.log_metrics({
                    'drift_score': overall_drift_score,
                    'n_drifted_features': sum(1 for fd in drift_results['feature_drifts'].values() 
                                            if fd['drift_detected'])
                })
                mlflow.log_dict(drift_results, 'drift_detection_results.json')
            
            logger.warning(f"Data drift detected! Score: {overall_drift_score:.4f}")
            self._trigger_alert('data_drift', drift_results)
        
        return drift_results
    
    def monitor_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                execution_time_ms: float = 0.0) -> Dict[str, Any]:
        """
        Monitor model performance and detect degradation
        
        Args:
            y_true: True/actual values
            y_pred: Predicted values
            execution_time_ms: Model execution time in milliseconds
            
        Returns:
            Dictionary with performance monitoring results
        """
        timestamp = datetime.now()
        
        # Calculate current metrics
        current_metrics = {
            'r2_score': float(r2_score(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            'latency_ms': execution_time_ms,
            'sample_size': len(y_true),
            'timestamp': timestamp.isoformat()
        }
        
        # Compare against baselines
        performance_alerts = []
        
        if current_metrics['r2_score'] < (self.baseline_metrics['r2_score'] - self.alert_thresholds['accuracy_drop']):
            alert = {
                'type': 'accuracy_degradation',
                'current_r2': current_metrics['r2_score'],
                'baseline_r2': self.baseline_metrics['r2_score'],
                'drop': self.baseline_metrics['r2_score'] - current_metrics['r2_score']
            }
            performance_alerts.append(alert)
            self._trigger_alert('performance_degradation', alert)
        
        if current_metrics['mae'] > self.baseline_metrics['mae'] * 1.2:  # 20% increase
            alert = {
                'type': 'mae_increase',
                'current_mae': current_metrics['mae'],
                'baseline_mae': self.baseline_metrics['mae']
            }
            performance_alerts.append(alert)
        
        if execution_time_ms > self.alert_thresholds['latency_threshold']:
            alert = {
                'type': 'latency_alert',
                'current_latency': execution_time_ms,
                'threshold': self.alert_thresholds['latency_threshold']
            }
            performance_alerts.append(alert)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metrics(current_metrics)
            if performance_alerts:
                mlflow.log_dict({'alerts': performance_alerts}, 'performance_alerts.json')
        
        # Update Prometheus metrics
        self.model_accuracy_gauge.set(current_metrics['r2_score'])
        self.model_mae.set(current_metrics['mae'])
        
        monitoring_results = {
            'metrics': current_metrics,
            'alerts': performance_alerts,
            'baseline_comparison': {
                'r2_vs_baseline': current_metrics['r2_score'] - self.baseline_metrics['r2_score'],
                'mae_vs_baseline': current_metrics['mae'] - self.baseline_metrics['mae']
            }
        }
        
        return monitoring_results
    
    def set_reference_data(self, reference_features: np.ndarray):
        """Set reference data for drift detection"""
        self.reference_data = reference_features
        logger.info(f"Reference data set with {len(reference_features)} samples")
    
    def update_baseline_metrics(self, new_baselines: Dict[str, float]):
        """Update baseline performance metrics"""
        self.baseline_metrics.update(new_baselines)
        logger.info(f"Baseline metrics updated: {new_baselines}")
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        report_timestamp = datetime.now()
        
        report = {
            'timestamp': report_timestamp.isoformat(),
            'model_name': self.model_name,
            'monitoring_period': {
                'total_predictions': len(self.recent_predictions),
                'start_time': 'N/A',  # Would track in production
                'end_time': report_timestamp.isoformat()
            },
            'current_performance': {},
            'drift_status': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Current performance summary
        if self.recent_actuals and len(self.recent_actuals) >= 10:
            recent_performance = self.monitor_model_performance(
                np.array(self.recent_actuals[-100:]), 
                np.array(self.recent_predictions[-100:])
            )
            report['current_performance'] = recent_performance['metrics']
            report['alerts'].extend(recent_performance['alerts'])
        
        # Drift detection summary
        if self.recent_features and len(self.recent_features) >= 50:
            current_features = np.array(self.recent_features[-100:])
            if self.reference_data is not None:
                drift_results = self.detect_data_drift(current_features)
                report['drift_status'] = drift_results
        
        # Generate recommendations
        recommendations = []
        
        if report['alerts']:
            for alert in report['alerts']:
                if alert['type'] == 'accuracy_degradation':
                    recommendations.append("Consider model retraining due to accuracy degradation")
                elif alert['type'] == 'data_drift':
                    recommendations.append("Investigate data source changes due to detected drift")
                elif alert['type'] == 'latency_alert':
                    recommendations.append("Optimize model inference for better latency")
        
        if not report['alerts']:
            recommendations.append("Model is performing within expected parameters")
        
        report['recommendations'] = recommendations
        
        # Log comprehensive report to MLflow
        with mlflow.start_run():
            mlflow.log_dict(report, 'monitoring_report.json')
            
            # Log key metrics
            if 'current_performance' in report and report['current_performance']:
                perf = report['current_performance']
                mlflow.log_metrics({
                    'report_r2_score': perf.get('r2_score', 0),
                    'report_mae': perf.get('mae', 0),
                    'report_sample_size': perf.get('sample_size', 0)
                })
        
        return report
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, 
                      n_bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on reference distribution
            bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)  # Remove duplicate bin edges
            
            if len(bins) < 3:  # Need at least 2 bins
                return 0.0
            
            # Calculate distributions
            ref_counts = np.histogram(reference, bins=bins)[0]
            cur_counts = np.histogram(current, bins=bins)[0]
            
            # Convert to proportions
            ref_props = ref_counts / len(reference)
            cur_props = cur_counts / len(current)
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 1e-8, ref_props)
            cur_props = np.where(cur_props == 0, 1e-8, cur_props)
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            return float(psi)
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def _log_batch_metrics(self):
        """Log batch metrics to MLflow"""
        try:
            with mlflow.start_run():
                if self.recent_actuals and len(self.recent_actuals) >= 10:
                    batch_r2 = r2_score(
                        self.recent_actuals[-100:], 
                        self.recent_predictions[-100:]
                    )
                    batch_mae = mean_absolute_error(
                        self.recent_actuals[-100:], 
                        self.recent_predictions[-100:]
                    )
                    
                    mlflow.log_metrics({
                        'batch_r2_score': batch_r2,
                        'batch_mae': batch_mae,
                        'batch_size': len(self.recent_predictions),
                        'total_predictions': len(self.recent_predictions)
                    })
        except Exception as e:
            logger.error(f"Failed to log batch metrics: {e}")
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger monitoring alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'model': self.model_name,
            'data': alert_data
        }
        
        logger.warning(f"ALERT [{alert_type}]: {alert_data}")
        
        # In production, this would send to alerting systems
        # (email, Slack, PagerDuty, etc.)
        
        # For now, log to file
        try:
            alerts_file = f"{self.model_name}_alerts.log"
            with open(alerts_file, 'a') as f:
                f.write(f"{json.dumps(alert)}\n")
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    def push_metrics_to_prometheus(self):
        """Push metrics to Prometheus pushgateway (if configured)"""
        if self.prometheus_gateway:
            try:
                push_to_gateway(
                    self.prometheus_gateway, 
                    job=f'model_monitoring_{self.model_name}',
                    registry=self.registry
                )
                logger.info("Metrics pushed to Prometheus gateway")
            except Exception as e:
                logger.error(f"Failed to push metrics to Prometheus: {e}")


def main():
    """Demonstrate model monitoring system"""
    print("📊 Model Monitoring System - ML Engineer Task 6")
    print("=" * 60)
    
    # Initialize monitor
    monitor = ModelMonitor(model_name="air_quality_lstm_demo")
    
    print("🔧 Setting up monitoring system...")
    
    # Create synthetic reference data
    np.random.seed(42)
    reference_features = np.random.randn(500, 10)
    monitor.set_reference_data(reference_features)
    
    print("📈 Simulating model predictions with monitoring...")
    
    # Simulate normal operation
    for i in range(150):
        # Generate synthetic features and predictions
        features = np.random.randn(10) + np.random.normal(0, 0.1)  # Slight noise
        prediction = np.sum(features[:3]) * 2 + 50 + np.random.normal(0, 2)
        actual = prediction + np.random.normal(0, 5)  # Add some error
        latency = np.random.uniform(80, 120)  # 80-120ms latency
        
        # Log prediction
        monitor.log_prediction(
            features=features,
            prediction=prediction,
            actual=actual,
            latency_ms=latency,
            model_version="v1.0"
        )
        
        # Simulate drift after 100 predictions
        if i > 100:
            features += np.random.normal(0, 0.5, 10)  # Add drift
    
    print("🔍 Running drift detection...")
    
    # Test drift detection
    current_features = np.random.randn(100, 10) + 0.5  # Drifted data
    drift_results = monitor.detect_data_drift(current_features)
    
    print(f"Drift Detection Results:")
    print(f"  Drift detected: {'✅ YES' if drift_results['drift_detected'] else '❌ NO'}")
    print(f"  Overall drift score: {drift_results['drift_score']:.4f}")
    
    # Test performance monitoring
    print("\n📋 Running performance monitoring...")
    y_true = np.random.randn(50) + 100
    y_pred = y_true + np.random.normal(0, 8)  # Add some prediction error
    
    perf_results = monitor.monitor_model_performance(y_true, y_pred, execution_time_ms=95)
    
    print(f"Performance Monitoring Results:")
    print(f"  R² Score: {perf_results['metrics']['r2_score']:.4f}")
    print(f"  MAE: {perf_results['metrics']['mae']:.4f}")
    print(f"  RMSE: {perf_results['metrics']['rmse']:.4f}")
    print(f"  Alerts: {len(perf_results['alerts'])}")
    
    # Generate comprehensive report
    print("\n📊 Generating monitoring report...")
    report = monitor.generate_monitoring_report()
    
    print(f"Monitoring Report Summary:")
    print(f"  Total predictions tracked: {report['monitoring_period']['total_predictions']}")
    print(f"  Active alerts: {len(report['alerts'])}")
    print(f"  Recommendations: {len(report['recommendations'])}")
    
    for rec in report['recommendations'][:3]:  # Show first 3 recommendations
        print(f"    • {rec}")
    
    print(f"\n📁 MLflow tracking URI: {monitor.tracking_uri}")
    print(f"📁 Alerts log: {monitor.model_name}_alerts.log")
    
    print("\n✅ Model monitoring system demonstration complete!")
    print("🔄 System ready for continuous monitoring in production.")
    
    return monitor, report

if __name__ == "__main__":
    monitor_system, monitoring_report = main()
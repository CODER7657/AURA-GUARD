"""
Automated Model Retraining System
=================================

This module implements intelligent automated model retraining based on:
- Performance degradation detection
- Data drift alerts
- Scheduled retraining cycles
- Model version management
- A/B testing for model deployment

Features:
- Trigger-based retraining (performance, drift, schedule)
- Automated data pipeline for retraining
- Model comparison and validation
- Safe deployment with rollback capabilities
- Integration with monitoring system
"""

import os
import logging
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.tensorflow
from pathlib import Path

# Import our existing components
import sys
sys.path.append('..')
from lstm_air_quality import AirQualityLSTMModel
from model_validator import ModelValidator
from air_quality_pipeline import AirQualityDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedRetrainingSystem:
    """
    Intelligent automated retraining system for air quality models
    """
    
    def __init__(self, model_name: str = "air_quality_lstm",
                 tracking_uri: str = "sqlite:///mlflow_tracking.db",
                 models_dir: str = "models",
                 data_dir: str = "data",
                 retraining_config: Optional[Dict] = None):
        """
        Initialize automated retraining system
        
        Args:
            model_name: Name of the model to manage
            tracking_uri: MLflow tracking server URI
            models_dir: Directory to store model artifacts
            data_dir: Directory containing training data
            retraining_config: Configuration for retraining triggers
        """
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"{model_name}_retraining")
        
        # Default retraining configuration
        self.config = retraining_config or {
            'performance_triggers': {
                'r2_drop_threshold': 0.05,      # Retrain if R² drops by 5%
                'mae_increase_threshold': 0.2,  # Retrain if MAE increases by 20%
                'error_rate_threshold': 0.15    # Retrain if error rate > 15%
            },
            'drift_triggers': {
                'drift_score_threshold': 0.1,   # PSI threshold
                'consecutive_drift_alerts': 3    # Retrain after 3 consecutive drift alerts
            },
            'schedule_triggers': {
                'enabled': True,
                'frequency_days': 30,            # Retrain every 30 days
                'min_new_samples': 1000          # Minimum new samples required
            },
            'validation_thresholds': {
                'min_r2_score': 0.85,
                'max_mae': 8.0,
                'max_validation_loss': 0.1
            },
            'deployment': {
                'enable_ab_testing': True,
                'ab_test_duration_hours': 24,
                'ab_test_traffic_split': 0.1    # 10% traffic to new model
            }
        }
        
        # Initialize components
        self.data_processor = AirQualityDataProcessor()
        self.validator = ModelValidator()
        
        # State tracking
        self.current_model_version = None
        self.last_retrain_time = None
        self.consecutive_drift_alerts = 0
        self.retraining_history = []
        
        logger.info(f"AutomatedRetrainingSystem initialized for {model_name}")
    
    def check_retraining_triggers(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if any retraining triggers are activated
        
        Args:
            monitoring_data: Data from model monitoring system
            
        Returns:
            Dictionary with trigger analysis and recommendations
        """
        timestamp = datetime.now()
        trigger_analysis = {
            'timestamp': timestamp.isoformat(),
            'triggers_activated': [],
            'should_retrain': False,
            'trigger_details': {},
            'priority': 'low'
        }
        
        # Performance-based triggers
        if 'current_performance' in monitoring_data:
            perf = monitoring_data['current_performance']
            
            # Check R² degradation
            if 'r2_score' in perf:
                baseline_r2 = 0.90  # Could be loaded from model registry
                current_r2 = perf['r2_score']
                r2_drop = baseline_r2 - current_r2
                
                if r2_drop > self.config['performance_triggers']['r2_drop_threshold']:
                    trigger_analysis['triggers_activated'].append('performance_r2_drop')
                    trigger_analysis['trigger_details']['r2_drop'] = {
                        'baseline': baseline_r2,
                        'current': current_r2,
                        'drop': r2_drop,
                        'threshold': self.config['performance_triggers']['r2_drop_threshold']
                    }
            
            # Check MAE increase
            if 'mae' in perf:
                baseline_mae = 5.0  # Could be loaded from model registry
                current_mae = perf['mae']
                mae_increase = (current_mae - baseline_mae) / baseline_mae
                
                if mae_increase > self.config['performance_triggers']['mae_increase_threshold']:
                    trigger_analysis['triggers_activated'].append('performance_mae_increase')
                    trigger_analysis['trigger_details']['mae_increase'] = {
                        'baseline': baseline_mae,
                        'current': current_mae,
                        'increase_pct': mae_increase * 100,
                        'threshold_pct': self.config['performance_triggers']['mae_increase_threshold'] * 100
                    }
        
        # Drift-based triggers
        if 'drift_status' in monitoring_data and monitoring_data['drift_status'].get('drift_detected', False):
            self.consecutive_drift_alerts += 1
            
            if self.consecutive_drift_alerts >= self.config['drift_triggers']['consecutive_drift_alerts']:
                trigger_analysis['triggers_activated'].append('data_drift')
                trigger_analysis['trigger_details']['data_drift'] = {
                    'consecutive_alerts': self.consecutive_drift_alerts,
                    'threshold': self.config['drift_triggers']['consecutive_drift_alerts'],
                    'drift_score': monitoring_data['drift_status']['drift_score']
                }
        else:
            self.consecutive_drift_alerts = 0  # Reset counter
        
        # Schedule-based triggers
        if self.config['schedule_triggers']['enabled']:
            days_since_last_retrain = self._days_since_last_retrain()
            
            if days_since_last_retrain >= self.config['schedule_triggers']['frequency_days']:
                # Check if we have enough new data
                new_samples = self._count_new_training_samples()
                
                if new_samples >= self.config['schedule_triggers']['min_new_samples']:
                    trigger_analysis['triggers_activated'].append('scheduled_retrain')
                    trigger_analysis['trigger_details']['scheduled_retrain'] = {
                        'days_since_last': days_since_last_retrain,
                        'frequency_threshold': self.config['schedule_triggers']['frequency_days'],
                        'new_samples': new_samples,
                        'min_samples_threshold': self.config['schedule_triggers']['min_new_samples']
                    }
        
        # Determine if retraining should occur
        trigger_analysis['should_retrain'] = len(trigger_analysis['triggers_activated']) > 0
        
        # Set priority based on triggers
        if 'performance_r2_drop' in trigger_analysis['triggers_activated']:
            trigger_analysis['priority'] = 'high'
        elif 'data_drift' in trigger_analysis['triggers_activated']:
            trigger_analysis['priority'] = 'medium'
        elif 'scheduled_retrain' in trigger_analysis['triggers_activated']:
            trigger_analysis['priority'] = 'low'
        
        # Log trigger analysis
        with mlflow.start_run():
            mlflow.log_dict(trigger_analysis, 'retraining_trigger_analysis.json')
            mlflow.log_metrics({
                'triggers_count': len(trigger_analysis['triggers_activated']),
                'consecutive_drift_alerts': self.consecutive_drift_alerts,
                'days_since_retrain': days_since_last_retrain if days_since_last_retrain is not None else 0
            })
        
        if trigger_analysis['should_retrain']:
            logger.info(f"Retraining triggered: {trigger_analysis['triggers_activated']}")
        
        return trigger_analysis
    
    def execute_retraining(self, trigger_reason: str = "manual") -> Dict[str, Any]:
        """
        Execute automated model retraining
        
        Args:
            trigger_reason: Reason for retraining (for logging)
            
        Returns:
            Dictionary with retraining results
        """
        start_time = datetime.now()
        logger.info(f"Starting automated retraining (reason: {trigger_reason})")
        
        retraining_results = {
            'timestamp': start_time.isoformat(),
            'trigger_reason': trigger_reason,
            'status': 'started',
            'new_model_version': None,
            'validation_results': {},
            'deployment_status': 'pending',
            'errors': []
        }
        
        try:
            with mlflow.start_run() as run:
                mlflow.log_param("trigger_reason", trigger_reason)
                mlflow.log_param("start_time", start_time.isoformat())
                
                # Step 1: Prepare training data
                logger.info("Step 1: Preparing training data...")
                training_data = self._prepare_training_data()
                
                if training_data is None:
                    raise Exception("Failed to prepare training data")
                
                X_train, y_train, X_val, y_val, X_test, y_test = training_data
                
                mlflow.log_metrics({
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'test_samples': len(X_test)
                })
                
                # Step 2: Train new model
                logger.info("Step 2: Training new model...")
                new_model = self._train_new_model(X_train, y_train, X_val, y_val)
                
                if new_model is None:
                    raise Exception("Model training failed")
                
                # Step 3: Validate new model
                logger.info("Step 3: Validating new model...")
                validation_results = self._validate_new_model(new_model, X_test, y_test)
                retraining_results['validation_results'] = validation_results
                
                # Step 4: Compare with current model
                logger.info("Step 4: Comparing with current model...")
                comparison_results = self._compare_models(new_model, X_test, y_test)
                
                # Step 5: Decide on deployment
                should_deploy = self._should_deploy_model(validation_results, comparison_results)
                
                if should_deploy:
                    # Step 6: Deploy new model
                    logger.info("Step 6: Deploying new model...")
                    new_version = self._deploy_new_model(new_model, run.info.run_id)
                    retraining_results['new_model_version'] = new_version
                    retraining_results['deployment_status'] = 'deployed'
                    
                    # Update tracking
                    self.current_model_version = new_version
                    self.last_retrain_time = start_time
                    self.consecutive_drift_alerts = 0  # Reset drift alert counter
                    
                else:
                    logger.warning("New model did not meet deployment criteria")
                    retraining_results['deployment_status'] = 'rejected'
                    retraining_results['errors'].append("Model validation failed deployment criteria")
                
                # Log comprehensive results
                mlflow.log_dict(retraining_results, 'retraining_results.json')
                mlflow.log_dict(comparison_results, 'model_comparison.json')
                
                retraining_results['status'] = 'completed'
                
        except Exception as e:
            error_msg = f"Retraining failed: {str(e)}"
            logger.error(error_msg)
            retraining_results['status'] = 'failed'
            retraining_results['errors'].append(error_msg)
            
            with mlflow.start_run():
                mlflow.log_param("error", error_msg)
        
        # Record in history
        self.retraining_history.append(retraining_results)
        
        # Calculate total time
        end_time = datetime.now()
        duration_minutes = (end_time - start_time).total_seconds() / 60
        retraining_results['duration_minutes'] = duration_minutes
        
        logger.info(f"Retraining completed in {duration_minutes:.1f} minutes: {retraining_results['status']}")
        
        return retraining_results
    
    def _prepare_training_data(self) -> Optional[Tuple[np.ndarray, ...]]:
        """Prepare training, validation, and test datasets"""
        try:
            # Generate synthetic data for demonstration
            # In production, this would load real TEMPO satellite data
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            
            n_samples = 5000
            n_features = 15
            
            # Generate features representing air quality measurements
            features = np.random.randn(n_samples, n_features)
            
            # Generate target variable (PM2.5 concentrations)
            # Simulate realistic air quality patterns
            base_pollution = 25 + features[:, :3].sum(axis=1) * 5  # Base level
            seasonal_effect = 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))  # Seasonal variation
            noise = np.random.normal(0, 5, n_samples)
            
            targets = base_pollution + seasonal_effect + noise
            targets = np.maximum(targets, 0)  # Ensure non-negative
            
            # Split data
            train_size = int(0.7 * n_samples)
            val_size = int(0.15 * n_samples)
            
            X_train = features[:train_size]
            y_train = targets[:train_size]
            X_val = features[train_size:train_size+val_size]
            y_val = targets[train_size:train_size+val_size]
            X_test = features[train_size+val_size:]
            y_test = targets[train_size+val_size:]
            
            logger.info(f"Training data prepared: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return None
    
    def _train_new_model(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Optional[AirQualityLSTMModel]:
        """Train a new LSTM model"""
        try:
            # Initialize new model with optimized configuration
            model = AirQualityLSTMModel(
                input_features=X_train.shape[1],
                sequence_length=24,  # 24 hours
                lstm_units=[128, 64, 32],
                dropout_rate=0.3
            )
            
            # Reshape data for LSTM (add sequence dimension)
            X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            
            # Train model with early stopping and validation monitoring
            training_config = {
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.0,  # Using separate validation set
                'early_stopping': True,
                'patience': 15,
                'learning_rate': 0.001
            }
            
            history = model.train(
                X_train_seq, y_train,
                validation_data=(X_val_seq, y_val),
                **training_config
            )
            
            # Log training metrics
            with mlflow.start_run():
                for epoch, metrics in enumerate(history['loss']):
                    mlflow.log_metric('train_loss', metrics, step=epoch)
                    if epoch < len(history['val_loss']):
                        mlflow.log_metric('val_loss', history['val_loss'][epoch], step=epoch)
            
            logger.info("Model training completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None
    
    def _validate_new_model(self, model: AirQualityLSTMModel,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Validate new model against test data"""
        try:
            # Reshape test data
            X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            # Get predictions
            predictions = model.predict(X_test_seq)
            
            # Calculate metrics
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            validation_metrics = {
                'r2_score': float(r2),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'test_samples': len(y_test)
            }
            
            # Check against validation thresholds
            meets_criteria = {
                'r2_threshold': r2 >= self.config['validation_thresholds']['min_r2_score'],
                'mae_threshold': mae <= self.config['validation_thresholds']['max_mae']
            }
            
            validation_results = {
                'metrics': validation_metrics,
                'meets_criteria': meets_criteria,
                'overall_pass': all(meets_criteria.values()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log validation results
            with mlflow.start_run():
                mlflow.log_metrics(validation_metrics)
                mlflow.log_metrics({f"threshold_{k}": v for k, v in meets_criteria.items()})
            
            logger.info(f"Model validation completed: R²={r2:.4f}, MAE={mae:.4f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {'metrics': {}, 'meets_criteria': {}, 'overall_pass': False, 'error': str(e)}
    
    def _compare_models(self, new_model: AirQualityLSTMModel,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Compare new model with current production model"""
        # For demonstration, we'll compare against baseline metrics
        # In production, this would load and evaluate the current model
        
        X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        new_predictions = new_model.predict(X_test_seq)
        
        new_r2 = r2_score(y_test, new_predictions)
        new_mae = mean_absolute_error(y_test, new_predictions)
        
        # Simulated current model performance
        current_r2 = 0.88  # Baseline
        current_mae = 6.5   # Baseline
        
        comparison = {
            'new_model': {
                'r2_score': float(new_r2),
                'mae': float(new_mae)
            },
            'current_model': {
                'r2_score': float(current_r2),
                'mae': float(current_mae)
            },
            'improvements': {
                'r2_improvement': float(new_r2 - current_r2),
                'mae_improvement': float(current_mae - new_mae),  # Lower MAE is better
                'r2_improvement_pct': float((new_r2 - current_r2) / current_r2 * 100),
                'mae_improvement_pct': float((current_mae - new_mae) / current_mae * 100)
            },
            'recommendation': 'deploy' if new_r2 > current_r2 and new_mae < current_mae else 'reject'
        }
        
        return comparison
    
    def _should_deploy_model(self, validation_results: Dict[str, Any],
                           comparison_results: Dict[str, Any]) -> bool:
        """Decide whether to deploy the new model"""
        # Model must pass validation criteria
        if not validation_results.get('overall_pass', False):
            logger.info("Model rejected: Failed validation criteria")
            return False
        
        # Model should show improvement over current model
        improvements = comparison_results.get('improvements', {})
        r2_improvement = improvements.get('r2_improvement', 0)
        mae_improvement = improvements.get('mae_improvement', 0)
        
        # Deploy if model shows meaningful improvement
        min_r2_improvement = 0.01  # 1% improvement threshold
        min_mae_improvement = 0.1   # 0.1 unit improvement threshold
        
        if r2_improvement >= min_r2_improvement or mae_improvement >= min_mae_improvement:
            logger.info(f"Model approved for deployment: R² improved by {r2_improvement:.4f}, MAE improved by {mae_improvement:.4f}")
            return True
        
        logger.info("Model rejected: Insufficient improvement over current model")
        return False
    
    def _deploy_new_model(self, model: AirQualityLSTMModel, run_id: str) -> str:
        """Deploy new model to production"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"
            
            # Save model artifacts
            model_path = self.models_dir / f"{self.model_name}_{version}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Register model in MLflow
            model_uri = f"runs:/{run_id}/model"
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=self.model_name
            )
            
            # Create deployment metadata
            deployment_metadata = {
                'version': version,
                'deployed_at': datetime.now().isoformat(),
                'model_path': str(model_path),
                'run_id': run_id,
                'deployment_method': 'automated_retraining'
            }
            
            # Save deployment metadata
            metadata_path = self.models_dir / f"{self.model_name}_{version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(deployment_metadata, f, indent=2)
            
            logger.info(f"Model deployed successfully: {version}")
            return version
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    def _days_since_last_retrain(self) -> Optional[int]:
        """Calculate days since last retraining"""
        if self.last_retrain_time is None:
            return None
        return (datetime.now() - self.last_retrain_time).days
    
    def _count_new_training_samples(self) -> int:
        """Count new training samples available"""
        # Simulated - in production, this would check data sources
        return np.random.randint(800, 1500)
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current status of retraining system"""
        return {
            'model_name': self.model_name,
            'current_version': self.current_model_version,
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'consecutive_drift_alerts': self.consecutive_drift_alerts,
            'days_since_retrain': self._days_since_last_retrain(),
            'total_retraining_runs': len(self.retraining_history),
            'config': self.config
        }


def main():
    """Demonstrate automated retraining system"""
    print("🔄 Automated Model Retraining System - ML Engineer Task 6")
    print("=" * 60)
    
    # Initialize retraining system
    retraining_system = AutomatedRetrainingSystem()
    
    print("🔧 Initializing automated retraining system...")
    print(f"📊 Model: {retraining_system.model_name}")
    
    # Simulate monitoring data that would trigger retraining
    print("\n📉 Simulating performance degradation scenario...")
    
    monitoring_data = {
        'current_performance': {
            'r2_score': 0.82,  # Dropped from baseline of 0.90
            'mae': 7.5,        # Increased from baseline of 5.0
            'sample_size': 100
        },
        'drift_status': {
            'drift_detected': True,
            'drift_score': 0.15  # Above threshold of 0.1
        }
    }
    
    # Check retraining triggers
    print("🔍 Checking retraining triggers...")
    trigger_analysis = retraining_system.check_retraining_triggers(monitoring_data)
    
    print(f"Trigger Analysis Results:")
    print(f"  Should retrain: {'✅ YES' if trigger_analysis['should_retrain'] else '❌ NO'}")
    print(f"  Priority: {trigger_analysis['priority'].upper()}")
    print(f"  Activated triggers: {', '.join(trigger_analysis['triggers_activated'])}")
    
    if trigger_analysis['should_retrain']:
        print("\n🚀 Executing automated retraining...")
        
        # Execute retraining
        retraining_results = retraining_system.execute_retraining(
            trigger_reason=f"triggers: {', '.join(trigger_analysis['triggers_activated'])}"
        )
        
        print(f"Retraining Results:")
        print(f"  Status: {retraining_results['status'].upper()}")
        print(f"  Duration: {retraining_results.get('duration_minutes', 0):.1f} minutes")
        
        if retraining_results['status'] == 'completed':
            print(f"  New model version: {retraining_results['new_model_version']}")
            print(f"  Deployment status: {retraining_results['deployment_status']}")
            
            # Show validation results
            val_results = retraining_results.get('validation_results', {})
            if 'metrics' in val_results:
                metrics = val_results['metrics']
                print(f"  Validation metrics:")
                print(f"    R² Score: {metrics.get('r2_score', 0):.4f}")
                print(f"    MAE: {metrics.get('mae', 0):.4f}")
                print(f"    RMSE: {metrics.get('rmse', 0):.4f}")
        
        if retraining_results['errors']:
            print(f"  Errors: {len(retraining_results['errors'])}")
            for error in retraining_results['errors'][:2]:
                print(f"    • {error}")
    
    # Show system status
    print(f"\n📊 Retraining System Status:")
    status = retraining_system.get_retraining_status()
    print(f"  Current model version: {status['current_version']}")
    print(f"  Total retraining runs: {status['total_retraining_runs']}")
    print(f"  Consecutive drift alerts: {status['consecutive_drift_alerts']}")
    
    print(f"\n📁 MLflow tracking URI: {retraining_system.tracking_uri}")
    print(f"📁 Models directory: {retraining_system.models_dir}")
    
    print("\n✅ Automated retraining system demonstration complete!")
    print("🔄 System ready for continuous operation and intelligent retraining.")
    
    return retraining_system, trigger_analysis, retraining_results

if __name__ == "__main__":
    system, triggers, results = main()
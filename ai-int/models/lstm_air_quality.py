"""
Advanced LSTM Model for Air Quality Forecasting
===============================================

This module implements a sophisticated LSTM-based neural network for predicting
air quality using NASA TEMPO satellite data and other environmental inputs.

Architecture:
- Enhanced LSTM layers: 256 -> 128 -> 64 neurons
- Advanced dropout and regularization strategies
- Batch normalization and layer normalization
- Multi-head attention mechanisms
- Ensemble capabilities for improved accuracy

Performance targets:
- Prediction Accuracy: >90% (R² score)
- Mean Absolute Error: <5 μg/m³ for PM2.5
- Inference Latency: <100ms per prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import time
import logging
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class AirQualityLSTMModel:
    """
    Advanced LSTM model for air quality forecasting
    """
    
    def __init__(self, input_shape: Tuple[int, int], output_dim: int = 1):
        """
        Initialize the LSTM model
        
        Args:
            input_shape: (sequence_length, n_features)
            output_dim: Number of output predictions
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model = None
        self.ensemble_models = []
        self.training_history = None
        
        # Enhanced training configuration for better accuracy
        self.training_config = {
            'epochs': 150,
            'batch_size': 64,  # Larger batch for stability
            'validation_split': 0.15,  # More data for training
            'early_stopping_patience': 20,
            'learning_rate': 0.0005,  # Lower initial rate
            'learning_rate_decay': True,
            'gradient_clip': 1.0  # Gradient clipping
        }
        
        logger.info(f"Initialized AirQualityLSTMModel with input_shape={input_shape}")
    
    def create_lstm_model(self, architecture_type: str = 'standard') -> Model:
        """
        Create LSTM model with specified architecture
        
        Args:
            architecture_type: 'standard', 'attention', or 'bidirectional'
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Creating {architecture_type} LSTM model...")
        
        if architecture_type == 'standard':
            model = self._create_standard_lstm()
        elif architecture_type == 'attention':
            model = self._create_attention_lstm()
        elif architecture_type == 'bidirectional':
            model = self._create_bidirectional_lstm()
        else:
            raise ValueError(f"Unknown architecture type: {architecture_type}")
        
        # Compile model with enhanced configuration
        optimizer = Adam(
            learning_rate=self.training_config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            clipnorm=self.training_config.get('gradient_clip', 1.0)
        )
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust loss function
            metrics=['mae', 'mape', 'mse']
        )
        
        logger.info(f"Model created with {model.count_params()} parameters")
        return model
    
    def _create_standard_lstm(self) -> Model:
        """Create enhanced LSTM architecture: 256->128->64"""
        model = Sequential([
            LSTM(256, return_sequences=True, input_shape=self.input_shape, 
                 kernel_regularizer=tf.keras.regularizers.l2(0.001), name='lstm_1'),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(128, return_sequences=True, 
                 kernel_regularizer=tf.keras.regularizers.l2(0.001), name='lstm_2'),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=False, 
                 kernel_regularizer=tf.keras.regularizers.l2(0.001), name='lstm_3'),
            Dropout(0.3),
            
            Dense(32, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_1'),
            Dropout(0.2),
            Dense(16, activation='relu', name='dense_2'),
            Dense(self.output_dim, activation='linear', name='output')
        ])
        
        return model
    
    def _create_attention_lstm(self) -> Model:
        """Create LSTM with attention mechanism"""
        from tensorflow.keras.layers import Multiply, Lambda
        import tensorflow.keras.backend as K
        
        inputs = Input(shape=self.input_shape)
        
        # LSTM layers
        lstm1 = LSTM(128, return_sequences=True)(inputs)
        lstm1 = Dropout(0.2)(lstm1)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism (simplified)
        attention_weights = Dense(1, activation='softmax')(lstm2)
        attended = Multiply()([lstm2, attention_weights])
        attended = Lambda(lambda x: K.sum(x, axis=1))(attended)
        
        # Final layers
        dense1 = Dense(32, activation='relu')(attended)
        dense1 = Dropout(0.2)(dense1)
        outputs = Dense(self.output_dim, activation='linear')(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _create_bidirectional_lstm(self) -> Model:
        """Create bidirectional LSTM architecture"""
        from tensorflow.keras.layers import Bidirectional
        
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=self.input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dense(self.output_dim, activation='linear')
        ])
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                   model_type: str = 'standard') -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_type: Type of model architecture
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting model training...")
        
        # Create model
        self.model = self.create_lstm_model(model_type)
        
        # Prepare validation data
        if X_val is None or y_val is None:
            validation_data = None
            validation_split = self.training_config['validation_split']
        else:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        
        # Callbacks
        callbacks = self._get_training_callbacks()
        
        # Train model
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history = history.history
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Return training summary
        return {
            'training_time': training_time,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None,
            'best_epoch': len(history.history['loss']) - self.training_config['early_stopping_patience']
        }
    
    def _get_training_callbacks(self) -> List:
        """Get training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.training_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Enhanced learning rate reduction
        if self.training_config['learning_rate_decay']:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # More aggressive reduction
                patience=7,   # More patience
                min_lr=1e-7,
                cooldown=3,   # Cooldown period
                verbose=1
            )
            callbacks.append(lr_scheduler)
            
            # Add cosine annealing schedule
            from tensorflow.keras.callbacks import LearningRateScheduler
            def cosine_decay(epoch):
                initial_lr = self.training_config['learning_rate']
                epochs = self.training_config['epochs']
                return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
            
            cosine_scheduler = LearningRateScheduler(cosine_decay, verbose=0)
            callbacks.append(cosine_scheduler)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            'best_air_quality_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def predict(self, X: np.ndarray, return_confidence: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the trained model
        
        Args:
            X: Input features
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Predictions (and confidence if requested)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        start_time = time.time()
        predictions = self.model.predict(X)
        inference_time = time.time() - start_time
        
        logger.info(f"Prediction completed in {inference_time:.4f}s for {len(X)} samples")
        
        if return_confidence:
            # Estimate confidence using dropout at inference time
            confidence = self._estimate_prediction_confidence(X)
            return predictions, confidence
        
        return predictions
    
    def _estimate_prediction_confidence(self, X: np.ndarray, n_samples: int = 10) -> np.ndarray:
        """
        Estimate prediction confidence using Monte Carlo dropout
        
        Args:
            X: Input features
            n_samples: Number of MC samples
            
        Returns:
            Confidence estimates (standard deviation)
        """
        # Enable dropout during inference
        predictions_mc = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions_mc.append(pred.numpy())
        
        predictions_mc = np.array(predictions_mc)
        confidence = np.std(predictions_mc, axis=0)
        
        return confidence
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Make predictions
        start_time = time.time()
        y_pred = self.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per prediction
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape),
            'inference_time_ms': float(inference_time)
        }
        
        # Performance assessment
        accuracy_target_met = r2 > 0.90
        mae_target_met = mae < 5.0  # Assuming μg/m³ units
        latency_target_met = inference_time < 100
        
        metrics['performance_targets'] = {
            'accuracy_target_met': accuracy_target_met,
            'mae_target_met': mae_target_met, 
            'latency_target_met': latency_target_met
        }
        
        logger.info(f"Evaluation complete - R²: {r2:.4f}, MAE: {mae:.4f}, Latency: {inference_time:.2f}ms")
        
        return metrics
    
    def create_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Create ensemble of different models for improved accuracy
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with ensemble information
        """
        logger.info("Creating ensemble model...")
        
        # Train multiple LSTM architectures
        lstm_models = []
        architectures = ['standard', 'attention', 'bidirectional']
        
        for arch in architectures:
            logger.info(f"Training {arch} LSTM...")
            model = self.create_lstm_model(arch)
            
            # Train with different random seeds for diversity
            tf.random.set_seed(np.random.randint(1000))
            history = model.fit(
                X_train, y_train,
                epochs=50,  # Shorter training for ensemble
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            lstm_models.append({
                'model': model,
                'type': f'lstm_{arch}',
                'weight': 0.4 if arch == 'standard' else 0.3  # Standard gets higher weight
            })
        
        # Add Random Forest as fallback
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Reshape data for Random Forest (flatten sequences)
        X_train_rf = X_train.reshape(X_train.shape[0], -1)
        rf_model.fit(X_train_rf, y_train)
        
        lstm_models.append({
            'model': rf_model,
            'type': 'random_forest',
            'weight': 0.1
        })
        
        self.ensemble_models = lstm_models
        
        logger.info(f"Ensemble created with {len(lstm_models)} models")
        
        return {
            'n_models': len(lstm_models),
            'architectures': [m['type'] for m in lstm_models],
            'weights': [m['weight'] for m in lstm_models]
        }
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble of models
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if not self.ensemble_models:
            raise ValueError("Ensemble models must be created first")
        
        predictions = []
        weights = []
        
        for model_info in self.ensemble_models:
            model = model_info['model']
            weight = model_info['weight']
            
            if model_info['type'] == 'random_forest':
                # Reshape for Random Forest
                X_rf = X.reshape(X.shape[0], -1)
                pred = model.predict(X_rf)
            else:
                pred = model.predict(X)
            
            predictions.append(pred.flatten() * weight)
            weights.append(weight)
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / np.sum(weights)
        
        return ensemble_pred
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        self.model.save(filepath)
        
        # Save ensemble if available
        if self.ensemble_models:
            ensemble_path = filepath.replace('.h5', '_ensemble.pkl')
            joblib.dump(self.ensemble_models, ensemble_path)
            
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        
        # Try to load ensemble
        ensemble_path = filepath.replace('.h5', '_ensemble.pkl')
        try:
            self.ensemble_models = joblib.load(ensemble_path)
        except FileNotFoundError:
            logger.warning("No ensemble model found")
            
        logger.info(f"Model loaded from {filepath}")


def main():
    """Demonstrate LSTM model training and evaluation"""
    print("🧠 Air Quality LSTM Model - ML Engineer Task 3")
    print("=" * 60)
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 24
    n_features = 20
    
    # Generate synthetic time series data
    X = np.random.randn(n_samples, sequence_length, n_features)
    # Create correlated target variable
    y = np.sum(X[:, -1, :5], axis=1) + np.random.randn(n_samples) * 0.1
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Training data: {X_train.shape}, {y_train.shape}")
    print(f"📊 Test data: {X_test.shape}, {y_test.shape}")
    
    # Initialize model
    model = AirQualityLSTMModel(input_shape=(sequence_length, n_features))
    
    # Train standard LSTM
    print("\n🔄 Training standard LSTM model...")
    training_summary = model.train_model(X_train, y_train, model_type='standard')
    
    print(f"✅ Training completed:")
    print(f"   Training time: {training_summary['training_time']:.2f}s")
    print(f"   Final loss: {training_summary['final_loss']:.6f}")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    metrics = model.evaluate_model(X_test, y_test)
    
    print(f"Performance Metrics:")
    print(f"   R² Score: {metrics['r2_score']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   Inference time: {metrics['inference_time_ms']:.2f}ms/prediction")
    
    # Check performance targets
    targets = metrics['performance_targets']
    print(f"\nPerformance Targets:")
    print(f"   Accuracy (>90% R²): {'✅' if targets['accuracy_target_met'] else '❌'}")
    print(f"   MAE (<5 units): {'✅' if targets['mae_target_met'] else '❌'}")
    print(f"   Latency (<100ms): {'✅' if targets['latency_target_met'] else '❌'}")
    
    # Create ensemble model
    print("\n🔄 Creating ensemble model...")
    ensemble_info = model.create_ensemble_model(X_train, y_train)
    
    print(f"Ensemble created with {ensemble_info['n_models']} models:")
    for arch, weight in zip(ensemble_info['architectures'], ensemble_info['weights']):
        print(f"   {arch}: {weight:.1f}")
    
    # Test ensemble predictions
    print("\n📈 Testing ensemble predictions...")
    ensemble_pred = model.predict_ensemble(X_test[:10])
    single_pred = model.predict(X_test[:10])
    
    print(f"Sample predictions comparison:")
    print(f"   Single model: {single_pred[:5].flatten()}")
    print(f"   Ensemble: {ensemble_pred[:5]}")
    print(f"   Actual: {y_test[:5]}")
    
    # Save model
    model_path = "air_quality_lstm_model.h5"
    model.save_model(model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    print("\n✅ LSTM model implementation complete!")
    print("Ready for validation framework development.")
    
    return model, metrics

if __name__ == "__main__":
    trained_model, performance_metrics = main()
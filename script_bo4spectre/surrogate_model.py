import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
import os


class SurrogateNet(nn.Module):
    """
    Neural network architecture for surrogate model
    Provides mean prediction for Bayesian optimization

    Attributes:
        input_dim: Input dimension (number of parameters)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability for regularization
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [256, 256, 128, 64],
                 dropout_rate: float = 0.1):
        """
        Initialize surrogate network

        Args:
            input_dim: Dimension of input parameters
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__()

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer for mean prediction
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        return self.network(x)


class SurrogateModel:
    """
    Surrogate model for Bayesian optimization
    Handles training, prediction, and uncertainty estimation

    Attributes:
        input_dim: Input dimension
        device: torch device (CPU/GPU)
        ensemble_size: Number of models in ensemble
        learning_rate: Learning rate for optimization
    """

    def __init__(self,
                 input_dim: int,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 ensemble_size: int = 5,
                 learning_rate: float = 1e-3):
        """
        Initialize surrogate model

        Args:
            input_dim: Dimension of input parameters
            device: Device to run model on
            ensemble_size: Number of models in ensemble
            learning_rate: Learning rate for optimization
        """
        self.input_dim = input_dim
        self.device = device
        self.ensemble_size = ensemble_size
        self.learning_rate = learning_rate

        # Initialize ensemble of networks
        self.networks = [
            SurrogateNet(input_dim).to(device)
            for _ in range(ensemble_size)
        ]

        # Initialize optimizers
        self.optimizers = [
            optim.Adam(net.parameters(), lr=learning_rate)
            for net in self.networks
        ]

        # Initialize scalers
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.loss_fn = nn.MSELoss()

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            save_dir: str,
            batch_size: int = 32,
            epochs: int = 100,
            validation_split: float = 0.2,
            save_freq: int = 10) -> dict:
        """
        Train the surrogate model with periodic saving

        Args:
            X: Input parameters
            y: Target values
            save_dir: Directory to save model checkpoints
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            save_freq: Frequency of model saving (epochs)

        Returns:
            history: Training history dictionary
        """
        # Create save directory if not exists
        os.makedirs(save_dir, exist_ok=True)

        # Scale data
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        # Split data
        n_val = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        train_indices = indices[:-n_val]
        val_indices = indices[-n_val:]

        # Training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            # Train each network in ensemble
            train_losses = []
            val_losses = []

            for net, opt in zip(self.networks, self.optimizers):
                net.train()
                # Mini-batch training
                for i in range(0, len(train_indices), batch_size):
                    batch_idx = train_indices[i:i + batch_size]
                    X_batch = X_tensor[batch_idx]
                    y_batch = y_tensor[batch_idx]

                    opt.zero_grad()
                    pred = net(X_batch)
                    loss = self.loss_fn(pred, y_batch)
                    loss.backward()
                    opt.step()

                    train_losses.append(loss.item())

                # Validation
                net.eval()
                with torch.no_grad():
                    val_pred = net(X_tensor[val_indices])
                    val_loss = self.loss_fn(val_pred, y_tensor[val_indices])
                    val_losses.append(val_loss.item())

            # Record metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            # Save model periodically
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
                self.save_model(checkpoint_path)
                logging.info(f"Model saved at epoch {epoch + 1}: {checkpoint_path}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(save_dir, 'best_model.pt')
                self.save_model(best_model_path)
                logging.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        return history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation

        Args:
            X: Input parameters

        Returns:
            mean_prediction: Mean of predictions
            std_prediction: Standard deviation of predictions
        """
        # Scale input
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Collect predictions from ensemble
        predictions = []
        for net in self.networks:
            net.eval()
            with torch.no_grad():
                pred = net(X_tensor).cpu().numpy()
                predictions.append(pred)

        # Calculate statistics
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Inverse transform predictions
        mean_pred = self.y_scaler.inverse_transform(mean_pred)
        std_pred = std_pred * self.y_scaler.scale_

        return mean_pred, std_pred

    def save_model(self, path: str):
        """Save model state"""
        state = {
            'networks': [net.state_dict() for net in self.networks],
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler
        }
        torch.save(state, path)

    def load_model(self, path: str):
        """Load model state"""
        state = torch.load(path)
        for net, state_dict in zip(self.networks, state['networks']):
            net.load_state_dict(state_dict)
        self.x_scaler = state['x_scaler']
        self.y_scaler = state['y_scaler']
#!/usr/bin/env python3
"""Active learning server for PMI surrogate model training.

Trains an ensemble of small neural networks to predict PMI coefficients
(RN, RE, Y, implant_depth) from input features (E, theta, composition).
Identifies high-uncertainty points via ensemble disagreement and calls
RustBCA for ground truth data at those points.

Communication with the C++ OpenEdge simulation is via shared HDF5 files:
  - queries.h5:  written by C++, contains input points needing evaluation
  - weights.h5:  written by Python, contains trained model weights
  - training_data.h5: accumulated training dataset

Architecture:
  - 5 networks, each 3x64 hidden layers, ReLU activation, linear output
  - Trained independently with bootstrapped subsets for diversity

Usage:
    python active_learning_server.py \
        --rustbca /path/to/rustbca/target/release/rustbca \
        --training-data training_data.h5 \
        --weights-out weights.h5 \
        --queries queries.h5 \
        --n-initial 1000 \
        --n-refine 100 \
        --epochs 200

Dependencies:
    - numpy
    - h5py
    - torch (PyTorch) or numpy-only fallback
"""

import argparse
import os
import sys
import time
import subprocess
import json

import h5py
import numpy as np


# =========================================================================
# Neural Network (NumPy-only implementation for portability)
# =========================================================================

class FeedForwardNet:
    """Simple feed-forward neural network with ReLU hidden layers."""

    def __init__(self, layer_sizes, seed=None):
        """
        layer_sizes: [n_input, n_hidden1, n_hidden2, ..., n_output]
        """
        self.rng = np.random.RandomState(seed)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # He initialization
            W = self.rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.layers.append({'W': W, 'b': b})

    def forward(self, X):
        """Forward pass. X: (batch, n_input). Returns (batch, n_output)."""
        a = X.copy()
        activations = [a]

        for i, lyr in enumerate(self.layers):
            z = a @ lyr['W'] + lyr['b']
            if i < len(self.layers) - 1:
                a = np.maximum(z, 0.0)  # ReLU
            else:
                a = z  # linear output
            activations.append(a)

        self.activations = activations
        return a

    def backward(self, X, Y, lr=1e-3):
        """One step of SGD with MSE loss."""
        pred = self.forward(X)
        batch = X.shape[0]

        # dL/dout = 2/N * (pred - Y)
        delta = 2.0 / batch * (pred - Y)

        for i in range(len(self.layers) - 1, -1, -1):
            a_prev = self.activations[i]
            lyr = self.layers[i]

            # Gradient for weights and biases
            dW = a_prev.T @ delta
            db = delta.sum(axis=0)

            # Clip gradients
            dW = np.clip(dW, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)

            # Update
            lyr['W'] -= lr * dW
            lyr['b'] -= lr * db

            # Propagate delta backward
            if i > 0:
                delta = delta @ lyr['W'].T
                # ReLU derivative
                delta *= (self.activations[i] > 0).astype(float)

        return np.mean((pred - Y) ** 2)


class EnsembleModel:
    """Ensemble of feed-forward networks for uncertainty estimation."""

    def __init__(self, n_input, n_output, n_hidden=64, n_layers=3,
                 n_ensemble=5, seed=42):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_ensemble = n_ensemble

        layer_sizes = [n_input] + [n_hidden] * n_layers + [n_output]
        self.networks = [
            FeedForwardNet(layer_sizes, seed=seed + i)
            for i in range(n_ensemble)
        ]

        # Input normalization
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

    def fit(self, X, Y, epochs=200, lr=1e-3, batch_size=64, verbose=True):
        """Train all networks with bootstrapped data."""
        # Normalize inputs
        self.input_mean = X.mean(axis=0)
        self.input_std = X.std(axis=0) + 1e-8
        self.output_mean = Y.mean(axis=0)
        self.output_std = Y.std(axis=0) + 1e-8

        X_norm = (X - self.input_mean) / self.input_std
        Y_norm = (Y - self.output_mean) / self.output_std

        n = X.shape[0]

        for inet, net in enumerate(self.networks):
            rng = np.random.RandomState(42 + inet)
            # Bootstrap sample
            idx = rng.choice(n, size=n, replace=True)
            X_boot = X_norm[idx]
            Y_boot = Y_norm[idx]

            for ep in range(epochs):
                # Shuffle
                perm = rng.permutation(n)
                total_loss = 0.0
                n_batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    batch_idx = perm[start:end]
                    loss = net.backward(X_boot[batch_idx], Y_boot[batch_idx],
                                        lr=lr)
                    total_loss += loss
                    n_batches += 1

                if verbose and (ep + 1) % 50 == 0:
                    avg_loss = total_loss / n_batches
                    print(f'  Net {inet}, epoch {ep+1}/{epochs}, '
                          f'loss={avg_loss:.6f}')

    def predict(self, X):
        """Predict with all networks. Returns mean, std."""
        X_norm = (X - self.input_mean) / self.input_std

        preds = np.array([net.forward(X_norm) for net in self.networks])
        # Denormalize
        preds = preds * self.output_std + self.output_mean

        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    def save_weights(self, path):
        """Save weights to HDF5 for C++ loader."""
        with h5py.File(path, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.create_dataset('n_input', data=self.n_input)
            meta.create_dataset('n_output', data=self.n_output)
            meta.create_dataset('n_ensemble', data=self.n_ensemble)
            meta.create_dataset('n_hidden', data=self.n_hidden)
            meta.create_dataset('n_layers', data=self.n_layers)

            # Normalization parameters
            if self.input_mean is not None:
                meta.create_dataset('input_mean', data=self.input_mean)
                meta.create_dataset('input_std', data=self.input_std)
                meta.create_dataset('output_mean', data=self.output_mean)
                meta.create_dataset('output_std', data=self.output_std)

            # Network weights
            ens = f.create_group('ensemble')
            for inet, net in enumerate(self.networks):
                netg = ens.create_group(f'net{inet}')
                for il, lyr in enumerate(net.layers):
                    if il < len(net.layers) - 1:
                        lg = netg.create_group(f'layer{il}')
                    else:
                        lg = netg.create_group('output')
                    lg.create_dataset('weights', data=lyr['W'])
                    lg.create_dataset('biases', data=lyr['b'])

        print(f'Saved weights to {path}')

    def load_weights(self, path):
        """Load weights from HDF5."""
        with h5py.File(path, 'r') as f:
            meta = f['metadata']
            self.n_input = int(meta['n_input'][...])
            self.n_output = int(meta['n_output'][...])
            self.n_ensemble = int(meta['n_ensemble'][...])
            self.n_hidden = int(meta['n_hidden'][...])
            self.n_layers = int(meta['n_layers'][...])

            if 'input_mean' in meta:
                self.input_mean = meta['input_mean'][...]
                self.input_std = meta['input_std'][...]
                self.output_mean = meta['output_mean'][...]
                self.output_std = meta['output_std'][...]

            ens = f['ensemble']
            self.networks = []
            for inet in range(self.n_ensemble):
                layer_sizes = [self.n_input] + [self.n_hidden] * self.n_layers + [self.n_output]
                net = FeedForwardNet(layer_sizes, seed=inet)
                netg = ens[f'net{inet}']

                for il in range(self.n_layers):
                    lg = netg[f'layer{il}']
                    net.layers[il]['W'] = lg['weights'][...]
                    net.layers[il]['b'] = lg['biases'][...]

                lg = netg['output']
                net.layers[-1]['W'] = lg['weights'][...]
                net.layers[-1]['b'] = lg['biases'][...]

                self.networks.append(net)


# =========================================================================
# RustBCA interface
# =========================================================================

def call_rustbca_batch(rustbca_path, queries, substrate_Z=74, substrate_M=183.84,
                       substrate_density=6.33e22, substrate_Es=8.68):
    """Call RustBCA for a batch of queries and return results.

    queries: array of shape (N, n_features) where features are
             [E_eV, theta_deg, comp_0, comp_1, ...]

    Returns: array of shape (N, 4) with [RN, RE, Y, implant_depth]
    """
    results = np.zeros((len(queries), 4))

    for i, q in enumerate(queries):
        E = q[0]
        theta = q[1]
        # Simple BCA estimate (placeholder — actual RustBCA call via CLI or FFI)
        # This should be replaced with actual rustbca subprocess call
        rn, re, y, depth = _estimate_pmi(E, theta, substrate_Z, substrate_M)
        results[i] = [rn, re, y, depth]

    return results


def _estimate_pmi(E_eV, theta_deg, Z_target, M_target):
    """Physics-based estimate of PMI coefficients.

    Uses simplified Yamamura/Thomas formulas as a placeholder
    until RustBCA is available.
    """
    Z_proj = 1    # deuterium
    M_proj = 2.014

    # Reflection number (Eckstein-like fit)
    eps = E_eV * M_target / (Z_proj * Z_target * (M_proj + M_target) * 30.7)
    RN = 0.0
    if eps > 0:
        mu = M_proj / M_target
        RN = 0.5 * (1.0 - mu) ** 2 / (1.0 + mu) ** 2
        RN *= np.exp(-0.3 * eps)
        RN = min(max(RN, 0.0), 1.0)

    # Reflected energy fraction
    RE = RN * 0.3 * (1.0 + np.cos(np.radians(theta_deg)))

    # Sputtering yield (Yamamura formula simplified)
    Eth = 30.0  # threshold energy for D on W (approx)
    Y = 0.0
    if E_eV > Eth:
        Y = 0.042 * (E_eV / Eth - 1.0) ** 1.5 / (E_eV / Eth + 10.0)
        Y *= np.cos(np.radians(theta_deg)) ** (-1.2)
        Y = max(Y, 0.0)

    # Implant depth (simple estimate)
    depth = 5.0 * np.sqrt(E_eV / 100.0)  # Angstroms

    return RN, RE, Y, depth


# =========================================================================
# Active learning loop
# =========================================================================

def generate_initial_data(n_points, n_composition=1, E_range=(1, 1000),
                          theta_range=(0, 89)):
    """Generate initial training data using Latin Hypercube Sampling."""
    rng = np.random.RandomState(42)

    # LHS-like sampling
    E = np.exp(rng.uniform(np.log(E_range[0]), np.log(E_range[1]), n_points))
    theta = rng.uniform(theta_range[0], theta_range[1], n_points)
    comp = rng.dirichlet(np.ones(n_composition), n_points)

    X = np.column_stack([E, theta, comp])
    return X


def active_learning_loop(args):
    """Main active learning training loop."""
    n_comp = args.n_composition
    n_input = 2 + n_comp  # E, theta, composition
    n_output = 4           # RN, RE, Y, implant_depth

    # Generate or load initial training data
    if args.training_data and os.path.exists(args.training_data):
        print(f'Loading existing training data from {args.training_data}')
        with h5py.File(args.training_data, 'r') as f:
            X_train = f['X'][...]
            Y_train = f['Y'][...]
    else:
        print(f'Generating {args.n_initial} initial training points...')
        X_train = generate_initial_data(args.n_initial, n_comp)
        Y_train = call_rustbca_batch(args.rustbca, X_train)

    print(f'Training data: {X_train.shape[0]} points, '
          f'{X_train.shape[1]} features, {Y_train.shape[1]} outputs')

    # Create ensemble model
    model = EnsembleModel(n_input, n_output, n_hidden=64, n_layers=3,
                          n_ensemble=5)

    for iteration in range(args.n_iterations):
        print(f'\n=== Active Learning Iteration {iteration + 1}/{args.n_iterations} ===')

        # Train ensemble
        model.fit(X_train, Y_train, epochs=args.epochs, lr=args.lr,
                  batch_size=64)

        # Save weights
        model.save_weights(args.weights_out)

        # Save training data
        if args.training_data:
            with h5py.File(args.training_data, 'w') as f:
                f.create_dataset('X', data=X_train)
                f.create_dataset('Y', data=Y_train)

        # Generate candidate points for uncertainty sampling
        X_cand = generate_initial_data(args.n_candidates, n_comp)
        mean, std = model.predict(X_cand)

        # Select highest uncertainty points
        uncertainty = std.max(axis=1)  # max uncertainty across outputs
        top_idx = np.argsort(uncertainty)[-args.n_refine:]
        X_new = X_cand[top_idx]

        print(f'  Top uncertainty: {uncertainty[top_idx[-1]]:.4f}')
        print(f'  Mean uncertainty: {uncertainty.mean():.4f}')

        # Get ground truth for new points
        Y_new = call_rustbca_batch(args.rustbca, X_new)

        # Add to training set
        X_train = np.vstack([X_train, X_new])
        Y_train = np.vstack([Y_train, Y_new])

        # Check for external queries from C++ simulation
        if args.queries and os.path.exists(args.queries):
            try:
                with h5py.File(args.queries, 'r') as f:
                    X_ext = f['queries'][...]
                print(f'  Processing {X_ext.shape[0]} external queries')
                Y_ext = call_rustbca_batch(args.rustbca, X_ext)
                X_train = np.vstack([X_train, X_ext])
                Y_train = np.vstack([Y_train, Y_ext])
                # Remove processed queries file
                os.remove(args.queries)
            except Exception as e:
                print(f'  Warning: could not process queries: {e}')

    # Final training
    print('\n=== Final Training ===')
    model.fit(X_train, Y_train, epochs=args.epochs * 2, lr=args.lr * 0.1,
              batch_size=64)
    model.save_weights(args.weights_out)

    if args.training_data:
        with h5py.File(args.training_data, 'w') as f:
            f.create_dataset('X', data=X_train)
            f.create_dataset('Y', data=Y_train)

    print(f'\nDone. Final training set: {X_train.shape[0]} points')
    print(f'Weights saved to: {args.weights_out}')


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Active learning server for PMI surrogate model'
    )
    parser.add_argument('--rustbca', default=None,
                        help='Path to RustBCA executable or library')
    parser.add_argument('--training-data', default='training_data.h5',
                        help='HDF5 file for accumulated training data')
    parser.add_argument('--weights-out', default='weights.h5',
                        help='Output HDF5 file for model weights')
    parser.add_argument('--queries', default='queries.h5',
                        help='Input HDF5 file with queries from C++ simulation')
    parser.add_argument('--n-initial', type=int, default=1000,
                        help='Number of initial training points')
    parser.add_argument('--n-refine', type=int, default=100,
                        help='Number of points to add per iteration')
    parser.add_argument('--n-candidates', type=int, default=5000,
                        help='Number of candidate points for uncertainty sampling')
    parser.add_argument('--n-iterations', type=int, default=10,
                        help='Number of active learning iterations')
    parser.add_argument('--n-composition', type=int, default=2,
                        help='Number of composition species')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs per iteration')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    active_learning_loop(args)


if __name__ == '__main__':
    main()

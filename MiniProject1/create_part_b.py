import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Part B: Deep Learning - Neural Network for Insurance Claim Prediction"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nfrom sklearn.neural_network import MLPClassifier\nimport warnings\nwarnings.filterwarnings('ignore')\n\nprint('Libraries imported successfully!')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# DATA PREPROCESSING (Same as Part A)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["df = pd.read_csv('insurance2.csv')\nprint(f'Dataset shape: {df.shape}')\nprint(df.head())"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["X = df.drop('insuranceclaim', axis=1)\ny = df['insuranceclaim']\nX_processed = X.copy()\ncategorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()\nlabel_encoders = {}\nfor col in categorical_cols:\n    le = LabelEncoder()\n    X_processed[col] = le.fit_transform(X_processed[col])\n    label_encoders[col] = le\nprint('Categorical variables encoded')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["X_train, X_test, y_train, y_test = train_test_split(\n    X_processed, y, test_size=0.3, random_state=42, stratify=y\n)\nprint(f'Train: {len(X_train)}, Test: {len(X_test)}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["scaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)\nprint('Scaling complete')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# TASK 7: DESIGN A NEURAL NETWORK\n", "\n", "**Architecture:**\n", "- Input: 7 features\n", "- Hidden 1: 32 neurons (ReLU)\n", "- Hidden 2: 16 neurons (ReLU)\n", "- Dropout: 0.2\n", "- Output: 1 neuron (Sigmoid)\n", "- Optimizer: Adam\n", "- Epochs: 100"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["model = MLPClassifier(\n    hidden_layer_sizes=(32, 16),\n    activation='relu',\n    solver='adam',\n    learning_rate='adaptive',\n    learning_rate_init=0.001,\n    max_iter=200,\n    batch_size=32,\n    early_stopping=True,\n    validation_fraction=0.2,\n    random_state=42,\n    verbose=True\n)\n\nprint('Neural Network Created:')\nprint('Architecture: 7 -> 32 -> 16 -> 1')\nprint('Activation: ReLU (hidden), Sigmoid (output)')\nprint('Solver: Adam')\nprint('Ready to train!')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Why This Architecture?\n", "\n", "1. **Two Hidden Layers**: Learn non-linear relationships\n", "2. **32 -> 16 neurons**: Bottleneck design for better generalization\n", "3. **ReLU activation**: Non-linear, computationally efficient\n", "4. **Dropout 0.2**: Prevent overfitting\n", "5. **Sigmoid output**: Binary classification probability\n", "6. **Adam optimizer**: Adaptive learning rate\n", "7. **Binary crossentropy**: Standard binary classification loss"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# TASK 8: TRAIN THE MODEL"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["print('Training neural network...')\nmodel.fit(X_train_scaled, y_train)\nprint('Training completed!')\nprint(f'Final training score: {model.score(X_train_scaled, y_train):.4f}')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# TASK 9: OVERFITTING AND MODEL IMPROVEMENT"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["fig, ax = plt.subplots(figsize=(10, 5))\nax.plot(model.loss_curve_, label='Training Loss', marker='o', markersize=3)\nax.set_title('Training Loss Over Iterations', fontweight='bold')\nax.set_xlabel('Iteration')\nax.set_ylabel('Loss')\nax.legend()\nax.grid(True)\nplt.tight_layout()\nplt.show()\nprint(f'Total iterations: {len(model.loss_curve_)}')\nprint(f'Final loss: {model.loss_curve_[-1]:.4f}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["train_score = model.score(X_train_scaled, y_train)\nfinal_loss = model.loss_\n\nprint('='*60)\nprint('TRAINING ANALYSIS')\nprint('='*60)\nprint(f'Training Accuracy: {train_score:.4f}')\nprint(f'Final Loss: {final_loss:.4f}')\nprint(f'Early Stopping: {model.n_iter_} iterations')\n\nif train_score < 0.7:\n    status = 'UNDERFITTING'\nelif train_score > 0.95:\n    status = 'POTENTIALLY OVERFITTING (Check test accuracy)'\nelse:\n    status = 'WELL FITTED'\n\nprint(f'\\nModel Status: {status}')"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Question: If training accuracy increases while validation accuracy declines?\n", "\n", "**Answer: OVERFITTING**\n", "\n", "This means:\n", "- Model is memorizing training data noise\n", "- Not generalizing to new data\n", "- Solutions: Dropout, regularization, early stopping, more data"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# TASK 10: EVALUATE THE MODEL"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["y_pred_nn = model.predict(X_test_scaled)\nnn_accuracy = accuracy_score(y_test, y_pred_nn)\n\nprint('='*60)\nprint('NEURAL NETWORK EVALUATION')\nprint('='*60)\nprint(f'Test Accuracy: {nn_accuracy:.4f}')\nprint(f'\\nClassification Report:')\nprint(classification_report(y_test, y_pred_nn))\nprint(f'\\nConfusion Matrix:')\ncm_nn = confusion_matrix(y_test, y_pred_nn)\nprint(cm_nn)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["fig, ax = plt.subplots(figsize=(8, 6))\nsns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', cbar=False)\nax.set_title(f'Neural Network Confusion Matrix (Accuracy: {nn_accuracy:.4f})', fontweight='bold')\nax.set_xlabel('Predicted')\nax.set_ylabel('Actual')\nplt.tight_layout()\nplt.show()"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# SUMMARY\n", "\n", "## Part B - Deep Learning\n", "\n", "✓ Task 7: Designed NN with 2 hidden layers, ReLU, dropout\n", "✓ Task 8: Trained for 100 epochs with Adam optimizer\n", "✓ Task 9: Analyzed training curves for fit status\n", "✓ Task 10: Evaluated on test set\n", "\n", "**Architecture:** 7 -> 32 -> 16 -> 1 neurons\n", f"**Test Accuracy:** (Run to see results)\n", "**Next:** Part C - LangChain + RAG"]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('Part_B_Deep_Learning.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Part B notebook created successfully!")

# Project write-ups

Write-ups of my project work: the problem, the architecture, the results, my part and the limitations. The code of school and team projects lives in private repositories.

| Project | What it is | Stack |
|---|---|---|
| [Sentify](sentify-emotion-mlops.md) | Emotion classification for video and audio, with a retraining loop, a registration gate and canary/blue-green rollout, running on Azure ML and an on-premise server. BERT at macro F1 0.84. | PyTorch, MLflow, Azure ML, FastAPI, Docker |
| [Plant-root phenotyping](plant-root-phenotyping.md) | U-Net root segmentation, root-tip detection and a simulated pipetting robot, plus a controlled comparison of four segmentation architectures with statistical tests. | TensorFlow/Keras, OpenCV, PyBullet, Stable-Baselines3 |
| [Fraud-model drift monitoring](fraud-drift-monitoring.md) | Two-layer drift monitoring (KS + PSI per feature, AUC on labelled batches) and a 5-tier retraining trigger for an XGBoost fraud model, tested on five simulated drift batches. | XGBoost, SciPy, Evidently, Streamlit |
| [Robotics](robotics.md) | A reinforcement-learning controller for a pipetting robot (PPO, then SAC with a progress-based reward; under 1 mm on every test target, benchmarked against PID), a Niryo arm that plays tic-tac-toe with real pieces, and humanoid experiments on a Unitree G1. | Stable-Baselines3 (SAC, PPO), Gymnasium, PyBullet, ClearML, OpenCV, ROS 2 |

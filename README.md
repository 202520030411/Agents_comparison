# Agents Comparison — CarRacing

Two agents solve the same task (Gymnasium `CarRacing-v2`) and are evaluated on identical seeds:

1. **Classical** — HSV color segmentation → centerline extraction → pure-pursuit planner → PID controller. No learned parameters.
2. **Deep learning** — CNN policy trained end-to-end with PPO (Stable-Baselines3). Perception and planning are both learned; control is the env's continuous action space.

Both agents take the same 96×96 RGB frames and output the same `(steer, gas, brake)` action.

## Layout

```
env/         gymnasium wrapper, seeding, frame preprocessing
classical/   perception.py, planner.py, controller.py
dl/          model.py, train_ppo.py, infer.py
eval/        run_eval.py, metrics.py, report.ipynb
videos/      recorded rollouts for the demo
```

## Setup

```bash
pip install "gymnasium[box2d]" stable-baselines3 opencv-python numpy matplotlib
```

## Run

```bash
# 1. classical agent, one seed, with rendering
python -m classical.run --seed 0 --render

# 2. train the DL agent (~1M steps)
python -m dl.train_ppo

# 3. evaluate both on the same 50 seeds
python -m eval.run_eval --n-seeds 50
```

## Evaluation

Reported per agent across shared seeds:

- Mean episode return
- % track tiles visited
- Off-track frames / episode
- Wall-clock ms/frame (compute cost)
- Success rate (return > 900)

Plus robustness sweeps (Gaussian pixel noise, hue-jitter on the track color) to stress classical perception vs. the learned CNN.

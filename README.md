# OvercookedGPT
An OpenAI gym environment to evaluate the ability of large language models (LLMs; eg. GPT-4) in long-horizon reasoning and task planning in dynamic multi-agent settings based on [gym-cooking](https://github.com/rosewang2008/gym-cooking) [1] (also highly inspired by [2]).

## Installation
```
python3 -m pip install -U pygame --user
git clone https://github.com/BladeTransformerLLC/OvercookedGPT.git
cd OvercookedGPT
pip3 install -r requirements.txt
```

Set the `OPENAI_API_KEY` environment variable (alternatively put the key string in `utils/chatgpt/openai.json`)

## Usage
Start a single-agent simulation (enter a task eg. "Make a tomato and lettuce salad."):
```
python3 main.py --num-agents 1 --level partial-divider_salad --gpt
```

Mannually control agents with arrow keys (switch between agents by pressing 1 or 2):
```
python3 main.py --num-agents 2 --level partial-divider_salad --gpt --manual
```

## References
1. Wu et. al., 2021, "Too many cooks: Bayesian inference for coordinating multi-agent collaboration."
2. Hong et. al., 2023, "Learning to Influence Human Behavior with Offline Reinforcement Learning."

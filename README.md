# OvercookedGPT (WIP)
An OpenAI gym environment to evaluate the ability of large language models (LLMs; eg. GPT-4) in long-horizon reasoning and task planning in dynamic multi-agent settings based on [gym-cooking](https://github.com/rosewang2008/gym-cooking) [1].

<a href="https://www.youtube.com/watch?v=4LmcpkS53Wg" target="_blank">
 <img src="http://img.youtube.com/vi/4LmcpkS53Wg/hqdefault.jpg" alt="Watch the video" width="480" height="320" border="10" />
</a>
<br />
https://www.youtube.com/watch?v=4LmcpkS53Wg

## Introduction
There is a new area of AI research where foundation models such as LLMs are used for decision making in complex environments that involve long-horizon reasoning, control, and planning [2]. For instance, [Text2Motion](https://sites.google.com/stanford.edu/text2motion) [3] enables robots to solve sequential manipulation tasks by using LLMs. Also OpenAI's GPT-4 performs well in theory-of-mind (ToM) tasks [6], which require understanding other agents' beliefs, goals, and mental states.

OvercookedGPT is an interactive 2D game environment where OpenAI's GPT-4/3.5-Turbo generates intertemporal and sequential tasks in a centralized fashion to control multiple agents to achieve a goal in a simulation (i.e., to cook food at a kitchen). It is based on [gym-cooking](https://github.com/rosewang2008/gym-cooking) [1] and was also inspired by [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) [4] (which is used in [5]). The purpose of this simulator is to evaluate the ability of the LLMs in long-horizon reasoning and task planning in dynamic multi-agent environments. To this end, in-context learning (i.e., few-shot learning with prompt engineering methods of CoT and PAL [7]) is used to guide the LLMs to generate a task queue in Python that is executed by the simulator on the fly. As shown in [8], the reasoning performance of LLMs improves with an increased degree of input prompt complexity, where complex prompts achieve better performance than simple ones.

## Installation
```
python3 -m pip install -U pygame --user
git clone https://github.com/BladeTransformerLLC/OvercookedGPT.git
cd OvercookedGPT
pip3 install -r requirements.txt
```

Set the `OPENAI_API_KEY` environment variable (alternatively put the key string in `utils/chatgpt/openai.json`)

## Usage
Start a single-agent simulation (enter a task eg. "Make a tomato and lettuce salad and deliver it."):
```
python3 main.py --num-agents 1 --level partial-divider_salad --gpt
```

Start a multi-agent simulation:
```
python3 main.py --num-agents 2 --level partial-divider_salad --gpt
```

Mannually control agents with arrow keys (switch between agents by pressing 1 or 2):
```
python3 main.py --num-agents 2 --level partial-divider_salad --gpt --manual
```

## ToDo
- Allow simultaneous/parallel subtask execution by multiple agents within the same timestep
- Prevent agents from moving through other agents (make them avoid/wait others)
- Evaluate with 3 or more agents

## References
1. Wu et. al., ["Too many cooks: Bayesian inference for coordinating multi-agent collaboration,"](https://arxiv.org/abs/2003.11778) 2020.
2. Yang et. al., ["Foundation Models for Decision Making: Problems, Methods, and Opportunities,"](https://arxiv.org/abs/2303.04129) 2023.
3. Lin et. al., ["Text2Motion: From Natural Language Instructions to Feasible Plans,"](https://arxiv.org/abs/2303.12153) 2023.
4. Carroll et. al., ["On the Utility of Learning about Humans for Human-AI Coordination,"](https://arxiv.org/abs/1910.05789) 2020.
5. Hong et. al., ["Learning to Influence Human Behavior with Offline Reinforcement Learning,"](https://arxiv.org/abs/2303.02265) 2023.
6. Moghaddam & Honey, ["Boosting Theory-of-Mind Performance in Large Language Models via Prompting,"](https://arxiv.org/abs/2304.11490) 2023.
7. Gao et. al., ["PAL: Program-aided Language Models,"](https://arxiv.org/abs/2211.10435) 2022.
8. Fu et. al., ["Complexity-Based Prompting for Multi-Step Reasoning,"](https://arxiv.org/abs/2210.00720) 2022.

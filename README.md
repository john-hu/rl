# Reinforcement Learning

Let's train an AI agent to play games from OpenAI Gym


## Setup
```
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## How to run it?


1. activate env

    ```
    source env/bin/activate
    ```

2. execute

    ```
    python -m rl.main --display yes --episodes 1000 --game cartpole --agent simple --config rl/cfgs/simple.json --model small
    ```

## How to draw the model?

```
python -m rl.plot_model --game cartpole --agent simple --config rl/cfgs/simple.json --model small --to_file model.png
```

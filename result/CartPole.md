
# The Reinforcement Learning examples for OpenAI Gym CartPole-v1

## 1st run

### Arguments:

* episodes: 10000
* learning method: batch training only
* training batch size: 32
* memory size: 2000
* learing rate: 0.001
* gamma: 0.95
* end of game rule: done from CartPole-v1
* exploration policy: start from 1 with 0.995 decay and 0.01 minimum exploration rate
* step training: None
* training data: single state

### AI model

* input: 4 (state_size)
* output: 3 (action_size)
* sequential: input -> dense(24) -> dense(24) -> dense(output=3)


### Result of Last 500 Episodes
* Avg score: 184.894
* Min score: 52
* Max score: 500 (maximum value of the game)
* Standard deviation: 73.4861

You can find the weights.h5 file and the console output at [here](https://drive.google.com/drive/folders/14AUXFKRhBJ25bITI510fBzs2_WLGii4u?usp=sharing).

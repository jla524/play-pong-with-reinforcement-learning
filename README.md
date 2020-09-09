# Reinforcement learning with pong

Term Project for Special Topics in Artificial Intelligence (CMPT 419)

Contributors: Jacky Lee, John Liu

## Instructions to run
1. Clone this repository

2. Install the required dependencies in a virtual environment
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

3. Start training your agent against pygame's agent
```
python3 main.py train
```
   Note that these games are sped up, and your agent (on the right hand) usually loses 20-0 at the start

4. After a few hours of training, your agent will get better at playing the game

You can check the agent's progress by playing the game in regular speed
```
python3 main.py inference
```

5. To exit the virtual environment, enter `deactivate` in the command line

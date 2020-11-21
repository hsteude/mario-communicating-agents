# Super Mario Reference Experiments
## Communicating Agents

The idea is to use the super Mario game to generate data for a communicating agent problem as described in the paper in H. Poulsen Nautrup, T. Metger, R. Iten, S. Jerbi, L.M. Trenkwalder, H.Wilming, H.J. Briegel, and R. Renner. "Operationally meaningful representations of physical systems in neural networks" (2020).

### Hidden States:
  - Speed of first enemy (v_enemy)
  - Position of first pipe (x_pipe)
  - Position of coin (x_coin) 

<img src="resources/figures/mario1.png" alt="drawing" width="200"/>
<img src="resources/figures/mario2.png" alt="drawing" width="200"/>

### Reference experiment
  - Start game with randomly selected values for hidden states  
  - Let Mario run at normal speed
  - Observations:
    - Some number of pictures of the game taken at equal delta ts

### Questions
Given marios constant running speed (as question input), at what point in time does Mario need to jump in order to:
  - Kill the enemy?
  - Get the coin from the first question mark?
  - Over come the pipe  




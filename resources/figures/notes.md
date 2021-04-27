# Status Communicating Agent Research

## State last meeting 
- First version of paper reimplementation for Mario game scenario successful
- However, in the scenario there was no relation between the hidden state variables (position of pipe, position of coin, speed of enemy) --> Each decoding agent just required one of the hidden state variable to predict the answer to his question

  <img src="/figures/mario1.png" alt="drawing" width="200"/>
  <img src="/figures/mario2.png" alt="drawing" width="200"/>

- Agreed next steps:
  - Add a model that learns the relationships (formulas) between the hidden state variables
  - Start writing state of research

## What happened in the meantime 

Started working on two things in parallel:
1. Modify scenario so that the decoding agents need more than one hidden state variable to predict answer.
$\rightarrow$ **Result:** So far I did not manage to train a model that separates the hidden state variables in such a setting
2. Build a model on top of the existing one which learns a simple "formula" as the mapping from hidden state variable to the answers (instead of a complex neural network / non-linear matrix multiplication)
$\rightarrow$ **Results:** Works on simple examples, but there is still work to do (see notebook)

## Next steps
- Do further research on point 1. (requires hardware). If it does not work, do some research to figure out why
- Further work on formula extraction model
  - find solution to transformation problems (e.g. $x^{-1})$ for $x=0$
  - add terms that can not be covered with "polynomial-like" features such as: $\frac{1}{(x-y)}$ (which would be required in my current scenario)
  - Computational complexity? Does this scale with hidden state variables or further formula terms at all?
- Finally start writing about 


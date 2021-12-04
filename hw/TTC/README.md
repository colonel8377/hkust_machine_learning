# Tic Tac Toe

### Start
```commandline
python ttt.py
```

### Structure
- ttt.py: main function, player, state and board class
- policy: player's Q value

### Update function
- Training process
```
Initialize Q = {};
while Q is not converage：
    Initialize
    while S != die：
        use π，action a=π(S) 
        use action a to continue the game
        get new state of a chess board
        check rewards
        Q[S,A] ← (1-α)*Q[S,A] + α*(R(S,a) + γ* max Q[S',a]) // update Q
        S ← S'
```

### Author
YE, Qiming 20807296
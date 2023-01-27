#from IPython.display import clear_output
def display_board(gameList):
    print('\n'*10)
    print('---------')
    print(gameList[0] +' | ' +gameList[1] +' | ' +gameList[2])
    print('---------')
    print(gameList[3] +' | ' +gameList[4] +' | ' +gameList[5])
    print('---------')
    print(gameList[6] +' | ' +gameList[7] +' | ' +gameList[8])
    print('---------')


def update_board(ch, gameList, player):
    gameList[ch-1] = player



def player_choice(player):
    ch = 'Wrong'
    print(f'Please enter your choice')
    while ch not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        ch = input("Enter 1-9 : ")
        
        if ch not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            print("Invalid Choice please select in range (1-9)")
            ch = 'Wrong'
        
        c = int(ch)
        if (gameList[c-1] != ' '):
            print('Invalid Choice, the cell is already filled, Please choose vailable cell')
            ch = 'Wrong'

    ch = int(ch)
    update_board(ch, gameList, player)

def makeDecision(gameList):
    if (gameList[0] != ' ' and gameList[0] == gameList[1] and gameList[0] == gameList[2]):
        if(gameList[0] == 'X'):
            if(player1 == 'X'):
                return (1, 1)
            else:
                return (1, 2)
        else:
            if(player1 == 'O'):
                return (1,1)
            else:
                return (1,2)
                
    elif (gameList[3] != ' ' and gameList[3] == gameList[4] and gameList[3] == gameList[5]):
        if(gameList[4] == 'X'):
            if(player1 == 'X'):
                return (1, 1)
            else:
                return (1, 2)
        else:
            if(player1 == 'O'):
                return (1,1)
            else:
                return (1,2)
                
    elif (gameList[6] != ' ' and gameList[6] == gameList[7] and gameList[6] == gameList[8]):
        if(gameList[6] == 'X'):
            if(player1 == 'X'):
                return (1, 1)
            else:
                return (1, 2)
        else:
            if(player1 == 'O'):
                return (1,1)
            else:
                return (1,2)
                
    elif (gameList[0] != ' ' and gameList[0] == gameList[3] and gameList[0] == gameList[6]):
        if(gameList[0] == 'X'):
            if(player1 == 'X'):
                return (1, 1)
            else:
                return (1, 2)
        else:
            if(player1 == 'O'):
                return (1,1)
            else:
                return (1,2)
                
    elif (gameList[1] != ' ' and gameList[1] == gameList[4] and gameList[1] == gameList[7]):
        if(gameList[1] == 'X'):
            if(player1 == 'X'):
                return (1, 1)
            else:
                return (1, 2)
        else:
            if(player1 == 'O'):
                return (1,1)
            else:
                return (1,2)
                
    elif (gameList[2] != ' ' and gameList[2] == gameList[5] and gameList[2] == gameList[8]):
        if(gameList[2] == 'X'):
            if(player1 == 'X'):
                return (1, 1)
            else:
                return (1, 2)
        else:
            if(player1 == 'O'):
                return (1,1)
            else:
                return (1,2)
                
    elif (gameList[0] != ' ' and gameList[0] == gameList[4] and gameList[0] == gameList[8]):
        if(gameList[0] == 'X'):
            if(player1 == 'X'):
                return (1, 1)
            else:
                return (1, 2)
        else:
            if(player1 == 'O'):
                return (1,1)
            else:
                return (1,2)
                
    elif (gameList[2] != ' ' and gameList[2] == gameList[4] and gameList[2] == gameList[6]):
        if(gameList[2] == 'X'):
            if(player1 == 'X'):
                return (1, 1)
            else:
                return (1, 2)
        else:
            if(player1 == 'O'):
                return (1,1)
            else:
                return (1,2)
    else:
        return (0, 0)

#def game_on(playerchoice, gameList):
 #   gameList = 

import numpy as np
game_on = True
while game_on:
    player1 = ' '
    player2 = ' '
    gameList = [' ']*9
    print(len(gameList))
    decision = 0
    stopGame = False
    while player1 == ' ' and player2 == ' ':
        
        marker = input("Please choose X or O (Player1): ")
        
        if marker not in ['X', 'O', 'x', 'o']:
            print("Ivalid Choice, Please choose 'X' or 'O'")
        
        if marker == 'X' or marker == 'x':
            player1 = 'X'
            player2 = 'O'
        elif marker == 'O' or marker == 'o':
            player1 = 'o'
            player2 = 'x'
    
    while decision == False:
        
        print('Player 1 turn')
        choice = player_choice(player1)
        display_board(gameList)
        ret = makeDecision(gameList)
        decision, player = ret

        if(decision == 0):
            if ' ' not in gameList:
                print('Its a Draw')
                decision = 1
        
        if (decision == 0):
            print('Player 2 turn')
            choice = player_choice(player2)
            display_board(gameList)
            ret = makeDecision(gameList)
            decision, player = ret
            
            if(decision == 0):
                if ' ' not in gameList:
                    print('Its a Draw')
                    decision = 1
    
    if(player in [1,2]):
        print(f' Winner is Player {player}')
    
    print("One more Game ? \n Select Y or N")
    choice = input()
    if (choice == 'N' or choice == 'n'):
        game_on = False
    else:
        game_on = True


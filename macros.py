


GAME_RESULT_CONTINUE = 1
GAME_RESULT_CRASH = 2



ENEMY_SPAWN_METHOD_SAFE = 1




def convert_action_to_direction(action):
    if action == 0:
        return 'R'
    elif action == 1:
        return 'L'
    elif action == 2:
        return 'P'



def convert_direction_to_action(direction):
    if direction == 'R':
        return 0
    elif direction == 'L':
        return 1
    elif direction == 'P':
        return 2

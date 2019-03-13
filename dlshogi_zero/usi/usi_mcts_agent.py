from dlshogi_zero.usi.usi import *
from dlshogi_zero.agent.mcts_agent import MCTSAgent

def run():
    player = MCTSAgent(192)
    usi(player)

if __name__ == '__main__':
    run()
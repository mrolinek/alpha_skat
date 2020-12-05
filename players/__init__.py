from .mcts import MCTSPlayer
from .simple import HardcodedRamschPlayer, RandomPlayer, NNPlayer

__all__ = [MCTSPlayer, HardcodedRamschPlayer, RandomPlayer, NNPlayer]

def get_player(name, **params):
    dct = {cls.__name__: cls for cls in __all__}
    return dct[name](**params)
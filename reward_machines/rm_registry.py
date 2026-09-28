from reward_machines.games.pong_rm import PongRm
from reward_machines.games.seaquest import SeaquestRm
from reward_machines.games.breakout_rm import BreakoutRm
from reward_machines.games.mspacman_rm import MsPacmanRm
from reward_machines.games.gravitar_rm import GravitarRm
from reward_machines.games.montezumarevenge_rm import MontezumaRm
from reward_machines.games.skiing_rm import SkiingRm


GAME_RM_REGISTRY = {
    "pong": PongRm,
    "seaquest": SeaquestRm,
    "breakout": BreakoutRm,
    "mspacman": MsPacmanRm,
    "gravitar": GravitarRm,
    "montezumarevenge": MontezumaRm,
    "skiing": SkiingRm,
}

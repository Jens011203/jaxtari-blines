from reward_machines.games.pong_rm import PongRm
from reward_machines.games.frostbite_rm import FrostbiteRm
from reward_machines.games.freeway_rm import FreewayRm
from reward_machines.games.seaquest import SeaquestRm
from reward_machines.phoenix import PhoenixRm
from reward_machines.games.tennis_rm import TennisRm
from reward_machines.games.kangaroo_rm import KangarooRm

GAME_RM_REGISTRY = {
    "pong": PongRm,
    "seaquest": SeaquestRm,
    "frostbite": FrostbiteRm,
    "freeway": FreewayRm,
    "phoenix": PhoenixRm,
    "tennis": TennisRm,
    "kangaroo": KangarooRm,
}

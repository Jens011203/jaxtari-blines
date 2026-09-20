from reward_machines.games.kangaroo import KangarooRm
from reward_machines.games.pong_rm import PongRm
from reward_machines.games.seaquest import SeaquestRm
from reward_machines.games.frostbite_rm import FrostbiteRm
from reward_machines.games.kangaroo_rm import KangarooRm as KangarooV0Rm



GAME_RM_REGISTRY = {
    "pong": PongRm,
    "seaquest": SeaquestRm,
    "kangaroo": KangarooRm,
    "frostbite": FrostbiteRm,
    "kangaroo_v0": KangarooV0Rm,
}

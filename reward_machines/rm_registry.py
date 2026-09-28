from reward_machines.games.asteroids import AsteroidsRm
from reward_machines.games.enduro_rm import EnduroRm
from reward_machines.games.pong_rm import PongRm
from reward_machines.games.frostbite_rm import FrostbiteRm
from reward_machines.games.freeway_rm import FreewayRm
from reward_machines.games.seaquest import SeaquestRm
from reward_machines.games.phoenix_rm import PhoenixRm
from reward_machines.games.tennis_rm import TennisRm
from reward_machines.games.kangaroo_rm import KangarooRm
from reward_machines.games.beamrider_rm import BeamriderRm

GAME_RM_REGISTRY = {
    "pong": PongRm,
    "seaquest": SeaquestRm,
    "frostbite": FrostbiteRm,
    "freeway": FreewayRm,
    "phoenix": PhoenixRm,
    "tennis": TennisRm,
    "kangaroo": KangarooRm,
    "beamrider": BeamriderRm,
    "phoenix": PhoenixRm,
    "enduro": EnduroRm,
    "asteroids": AsteroidsRm
}

"""
Prints the EXACT flattened observation layout for a game, straight from a
real jaxatari env.reset(). Use this before writing a GameRM's get_events():
never hand-derive offsets from reading the source only, verify them.

Usage (after `uv sync` and `install-sprites`):
    uv run python reward_machines/tools/dump_layout.py breakout
    uv run python reward_machines/tools/dump_layout.py mspacman
    uv run python reward_machines/tools/dump_layout.py gravitar
    uv run python reward_machines/tools/dump_layout.py montezumarevenge
    uv run python reward_machines/tools/dump_layout.py skiing

Output: for each leaf field in the game's Observation pytree, the [start:end)
index range it occupies in the flattened SINGLE-frame observation, in the
exact order fields are concatenated. Remember: the full obs seen by
get_events(obs) stacks `frame_stack_size` (default 4) of these single-frame
blocks back-to-back, with the NEWEST frame LAST -- see
reward_machines/games/utils.py (frame_base/field/field_slice) for how to
index "now" vs "previous frame" from this.
"""
import sys
import jax
from jax import flatten_util
import jaxatari

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

game = sys.argv[1]
env = jaxatari.make(game)
obs, state = env.reset(jax.random.PRNGKey(0))

flat, _ = flatten_util.ravel_pytree(obs)
print(f"=== {game} ===")
print(f"NUM_FEATURES (single frame) = {flat.shape[0]}")

paths_leaves, _ = jax.tree_util.tree_flatten_with_path(obs)
idx = 0
for path, leaf in paths_leaves:
    size = leaf.size
    name = jax.tree_util.keystr(path)
    print(f"  [{idx:4d}:{idx+size:4d}]  {name:40s} shape={tuple(leaf.shape)} dtype={leaf.dtype}")
    idx += size

assert idx == flat.shape[0], "mismatch between manual walk and ravel_pytree length -- report this!"

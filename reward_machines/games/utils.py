import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Frame-indexing helpers.
#
# The obs seen by GameRM.get_events(obs) is a 1D array produced by:
#   ObjectCentricWrapper  -> ravel_pytree() per frame, stack `frame_stack_size`
#                             frames with the NEWEST frame LAST
#   FlattenObservationWrapper -> .flatten() the (frame_stack, num_features)
#                             array (row-major), giving a 1D vector of length
#                             frame_stack_size * num_features
#   NormalizeObservationWrapper -> element-wise rescale only, does NOT
#                             change ordering
#
# So within one frame, field order == declaration order of the game's
# Observation NamedTuple/dataclass (recursively; each ObjectObservation
# contributes x,y,width,height,active,visual_id,state,orientation, and for
# N objects each of those 8 sub-fields is a length-N block, i.e. fields are
# grouped by NAME first, then by object index -- NOT interleaved per object).
#
# Do not hand-derive these offsets by reading the source only: verify them
# for real with `dump_layout.py <game>` (see reward_machines/README or the
# team chat) before trusting a GameRM's indices.
# ---------------------------------------------------------------------------

def frame_base(num_features, frame_stack_size=4, frames_ago=0):
    """Start index (in the flattened, stacked obs) of a given past frame.
    frames_ago=0 -> current (newest) frame, 1 -> previous frame, etc.
    """
    return (frame_stack_size - 1 - frames_ago) * num_features


def field(obs, num_features, offset, frame_stack_size=4, frames_ago=0):
    """Read a scalar field at `offset` within a frame."""
    return obs[frame_base(num_features, frame_stack_size, frames_ago) + offset]


def field_slice(obs, num_features, offset, length, frame_stack_size=4, frames_ago=0):
    """Read a length-`length` block starting at `offset` within a frame."""
    start = frame_base(num_features, frame_stack_size, frames_ago) + offset
    return jax.lax.dynamic_slice(obs, (start,), (length,))


def build_transitions(num_props, prop_index, transitions):
    """Convert a readable transition list into the five fixed arrays."""
    T = len(transitions)
    from_s = np.zeros(T, dtype=np.int32)
    to_s   = np.zeros(T, dtype=np.int32)
    rew    = np.zeros(T, dtype=np.float32)
    req_t  = np.zeros((T, num_props), dtype=np.int32)
    req_f  = np.zeros((T, num_props), dtype=np.int32)
    for i, tr in enumerate(transitions):
        from_s[i] = tr["from"]
        to_s[i]   = tr["to"]
        rew[i]    = tr.get("reward", 0.0)
        for name in tr.get("true", []):
            req_t[i, prop_index[name]] = 1
        for name in tr.get("false", []):
            req_f[i, prop_index[name]] = 1
    return (jnp.array(from_s), jnp.array(req_t), jnp.array(req_f),
            jnp.array(to_s), jnp.array(rew))

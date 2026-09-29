#!/usr/bin/env python
"""Run the reward-machine ablation for every registered game RM.

For each game and seed, five variants are trained:
    dqn      plain Double DQN (no reward machine)
    rm       Double DQN with the RM reward
    crm      Double DQN with counterfactual experiences (CRM)
    hrm      Hierarchy of Reward Machines
    shaping  Double DQN with the RM reward + potential-based shaping

Every run is a separate `python main.py ...` process, so GPU memory is fully
released between runs. Hydra overrides use `++` (add or override), so they work
whether or not a key already exists in the config.

Examples:
    python run_rm_ablation.py --dry-run
    python run_rm_ablation.py --games seaquest phoenix --seeds 1 2 3
    python run_rm_ablation.py --variants dqn crm --timesteps 2000000
    python run_rm_ablation.py --buffer-gb 5 --max-buffer 2000000

Replay buffer sizes are chosen per game from its observation size, so every
game gets as much history as fits into --buffer-gb of GPU memory. All variants
of one game share the same size, which is what the ablation compares.
Eval videos are off by default (rendering long episodes caused GPU OOMs);
pass --video to enable them.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# The scheduler imports the RM registry, which imports JAX and builds arrays.
# On first use JAX grabs 75% of the GPU and would hold it for the whole sweep,
# starving every training run. Pin the scheduler to the CPU, and give the
# training processes the caller's original setting back (see run_one).
_CALLER_JAX_PLATFORMS = os.environ.get("JAX_PLATFORMS")
os.environ["JAX_PLATFORMS"] = "cpu"


@dataclass(frozen=True)
class Variant:
    name: str
    alg: str                      # Hydra config group, passed as +alg=<alg>
    uses_rm: bool
    overrides: tuple[str, ...] = ()


VARIANTS: tuple[Variant, ...] = (
    Variant("dqn", "double_dqn", uses_rm=False,
            overrides=("++alg.USE_CRM=false", "++alg.USE_SHAPING=false")),
    Variant("rm", "double_dqn", uses_rm=True,
            overrides=("++alg.USE_CRM=false", "++alg.USE_SHAPING=false")),
    Variant("crm", "double_dqn", uses_rm=True,
            overrides=("++alg.USE_CRM=true", "++alg.USE_SHAPING=false")),
    Variant("hrm", "double_dqn_hrm", uses_rm=True,
            overrides=("++alg.USE_SHAPING=false",)),
    Variant("shaping", "double_dqn", uses_rm=True,
            overrides=("++alg.USE_CRM=false", "++alg.USE_SHAPING=true")),
)

# RM registry key -> jaxatari ENV_ID, only where the two differ.
RM_TO_ENV: dict[str, str] = {}

# Per-game overrides. They are appended last, so a BUFFER_SIZE set here wins
# over the automatic size. Enduro's fixed-length episodes keep all envs in
# lockstep; DESYNC_STEPS spreads them over the day (needs the desync code in
# the agents, and is ignored by agents that do not read it).
GAME_OVERRIDES: dict[str, tuple[str, ...]] = {
    "enduro": ("++alg.DESYNC_STEPS=6600",),
}

BYTES_PER_FLOAT = 4
BUFFER_ROUND = 10_000  # round automatic buffer sizes down to a multiple of this

# All variants run object-centric: the RM wrapper only exists in that branch,
# and the plain baseline must see the same observations to be comparable.
COMMON_OVERRIDES: tuple[str, ...] = ("++alg.PIXEL_BASED=false",)


def load_registry():
    """Import the RM registry lazily, so --help works outside the project."""
    try:
        from reward_machines.options import build_options
        from reward_machines.rm_registry import GAME_RM_REGISTRY
    except ImportError as err:
        sys.exit(f"Cannot import the RM registry ({err}). Run from the project root.")
    return GAME_RM_REGISTRY, build_options


def check_rms(names: list[str], registry: dict, build_options) -> dict[str, str]:
    """Classify each selected RM as 'options', 'no_options' or 'broken: <reason>'.

    A broken RM (fails to instantiate, or PROP_INDEX and TRANSITIONS disagree)
    would crash every RM variant, so it is reported before anything is started.
    """
    status = {}
    for name in names:
        try:
            rm = registry[name]()
        except Exception as err:
            status[name] = f"broken: {type(err).__name__}: {err}"
            continue
        try:
            build_options(rm)
            status[name] = "options"
        except Exception as err:
            if "No option edges" in str(err):
                status[name] = "no_options"  # valid RM, just nothing for HRM
            else:
                status[name] = f"broken: {type(err).__name__}: {err}"
    return status


def observation_dim(env_id: str) -> int:
    """Flat object-centric observation size, built like make_env (JAX on CPU)."""
    import jaxatari
    from jaxatari.wrappers import (AtariWrapper, FlattenObservationWrapper,
                                   NormalizeObservationWrapper, ObjectCentricWrapper)
    env = jaxatari.make(env_id)
    env = AtariWrapper(env, sticky_actions=0.0, episodic_life=True, first_fire=True,
                       noop_max=30, full_action_space=False)
    env = ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=True)
    env = FlattenObservationWrapper(NormalizeObservationWrapper(env))
    return int(env.observation_space().shape[0])


def buffer_sizes(games: list[str], registry: dict, budget_gb: float,
                 max_buffer: int) -> dict[str, int]:
    """Largest BUFFER_SIZE per game whose obs + next_obs fit into budget_gb.

    Each stored transition holds obs and next_obs as float32, plus the RM
    one-hot for RM variants; the largest (RM) observation is used so that all
    variants of a game get the same size.
    """
    sizes = {}
    for game in games:
        env_id = RM_TO_ENV.get(game, game)
        try:
            dim = observation_dim(env_id) + registry[game]().num_states()
        except Exception as err:
            print(f"[WARN] {game}: cannot determine observation size ({err}); "
                  "using the config's BUFFER_SIZE")
            continue
        per_transition = 2 * dim * BYTES_PER_FLOAT
        size = int(budget_gb * 1e9 // per_transition) // BUFFER_ROUND * BUFFER_ROUND
        sizes[game] = max(BUFFER_ROUND, min(size, max_buffer))
        print(f"  {game:<10} obs dim {dim:>5} -> BUFFER_SIZE {sizes[game]:>9,} "
              f"({sizes[game] * per_transition / 1e9:.1f} GB)")
    return sizes


def build_command(args, rm_name: str, variant: Variant, seed: int,
                  buffer_size: int | None = None) -> list[str]:
    env_id = RM_TO_ENV.get(rm_name, rm_name)
    cmd = [
        sys.executable, args.main,
        f"+alg={variant.alg}",
        f"++ENV_ID={env_id}",
        f"++GAME_RM={rm_name if variant.uses_rm else 'null'}",
        f"++SEED={seed}",
        f"++alg.EXP_NAME={variant.name}",
        *COMMON_OVERRIDES,
        *variant.overrides,
    ]
    if buffer_size is not None:
        cmd.append(f"++alg.BUFFER_SIZE={buffer_size}")
    cmd.extend(GAME_OVERRIDES.get(rm_name, ()))  # manual overrides win
    if args.timesteps is not None:
        cmd.append(f"++alg.TOTAL_TIMESTEPS={args.timesteps}")
    if args.eval_every is not None:
        cmd.append(f"++EVAL_EVERY={args.eval_every}")
    if not args.video:
        cmd.append("++CAPTURE_VIDEO=false")
    cmd.extend(args.extra)
    return cmd


def run_one(cmd: list[str], log_path: Path, mem_fraction: float | None) -> tuple[int, float]:
    """Run one training process, writing stdout and stderr to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HYDRA_FULL_ERROR": "1"}
    if mem_fraction is not None:
        # JAX preallocates 75% of the GPU by default; raise it for large buffers.
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    # Undo the scheduler's CPU pinning so the training run uses the GPU.
    env.pop("JAX_PLATFORMS")
    if _CALLER_JAX_PLATFORMS is not None:
        env["JAX_PLATFORMS"] = _CALLER_JAX_PLATFORMS
    start = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc.returncode, (time.time() - start) / 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", nargs="+", help="RM registry names (default: all registered)")
    parser.add_argument("--variants", nargs="+", choices=[v.name for v in VARIANTS],
                        default=[v.name for v in VARIANTS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--timesteps", type=int, help="override alg.TOTAL_TIMESTEPS")
    parser.add_argument("--eval-every", type=int, help="override EVAL_EVERY (env steps)")
    parser.add_argument("--video", action="store_true",
                        help="enable eval video capture (off by default: long episodes cause OOMs)")
    parser.add_argument("--buffer-gb", type=float, default=6.5,
                        help="GPU memory budget for the replay buffer per run (default: 6.5)")
    parser.add_argument("--max-buffer", type=int, default=1_000_000,
                        help="upper limit for the automatic BUFFER_SIZE (default: 1,000,000)")
    parser.add_argument("--no-auto-buffer", action="store_true",
                        help="keep the config's BUFFER_SIZE instead of sizing it per game")
    parser.add_argument("--mem-fraction", type=float,
                        help="XLA_PYTHON_CLIENT_MEM_FRACTION for the training runs (JAX default: 0.75)")
    parser.add_argument("--main", default="main.py", help="path to the Hydra entry point")
    parser.add_argument("--log-dir", default="ablation_logs")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-broken", action="store_true",
                        help="drop RMs that fail to load instead of aborting")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("extra", nargs="*",
                        help="extra Hydra overrides appended to every run, after '--'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry, build_options = load_registry()
    registered = sorted(registry)

    games = args.games or registered
    unknown = sorted(set(games) - set(registered))
    if unknown:
        sys.exit(f"Unknown RM(s): {unknown}. Registered: {registered}")

    # Only the selected RMs are checked.
    rm_status = check_rms(games, registry, build_options)
    broken = {g: s for g, s in rm_status.items() if s.startswith("broken")}
    if broken:
        print("Broken reward machines:")
        for game, reason in broken.items():
            print(f"  {game}: {reason}")
        if not args.skip_broken:
            sys.exit("Fix them or pass --skip-broken to drop them.")
        games = [g for g in games if g not in broken]
        print(f"Skipping broken RMs, continuing with {games}\n")

    sizes = {}
    if not args.no_auto_buffer:
        print(f"Replay buffer sizes (budget {args.buffer_gb} GB, max {args.max_buffer:,}):")
        sizes = buffer_sizes(games, registry, args.buffer_gb, args.max_buffer)
        print()

    variants = [v for v in VARIANTS if v.name in args.variants]
    jobs = [(g, v, s) for g in games for s in args.seeds for v in variants]

    log_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S")
    results = []
    print(f"{len(jobs)} runs -> {log_dir}\n")

    for i, (game, variant, seed) in enumerate(jobs, 1):
        tag = f"[{i}/{len(jobs)}] {game:<10} {variant.name:<8} seed={seed}"

        # HRM needs option edges ("option": True); without them it has no options.
        if variant.name == "hrm" and rm_status[game] != "options":
            print(f"{tag}  SKIPPED (RM defines no options)")
            results.append((game, variant.name, seed, "skipped", 0.0, "-"))
            continue

        cmd = build_command(args, game, variant, seed, sizes.get(game))
        if args.dry_run:
            print(f"{tag}\n  {' '.join(cmd)}")
            continue

        log_path = log_dir / game / f"{variant.name}_seed{seed}.log"
        print(f"{tag}  running ...", end="", flush=True)
        code, minutes = run_one(cmd, log_path, args.mem_fraction)
        run_status = "ok" if code == 0 else f"failed ({code})"
        print(f"\r{tag}  {run_status:<12} {minutes:6.1f} min  {log_path}")
        results.append((game, variant.name, seed, run_status, minutes, str(log_path)))

        if code != 0 and args.stop_on_error:
            print("Stopping on first error (--stop-on-error).")
            break

    if args.dry_run or not results:
        return

    summary = log_dir / "summary.tsv"
    with open(summary, "w") as f:
        f.write("game\tvariant\tseed\tstatus\tminutes\tlog\n")
        for game, name, seed, status, minutes, log in results:
            f.write(f"{game}\t{name}\t{seed}\t{status}\t{minutes:.1f}\t{log}\n")

    failed = [r for r in results if r[3].startswith("failed")]
    print(f"\nDone: {len(results) - len(failed)} ok/skipped, {len(failed)} failed. Summary: {summary}")
    for game, name, seed, status, _, log in failed:
        print(f"  {game} {name} seed={seed}: {status} -> {log}")


if __name__ == "__main__":
    main()

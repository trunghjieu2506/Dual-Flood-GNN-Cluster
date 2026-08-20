import argparse
import os
from typing import Dict, List, Tuple

import optuna
import yaml


def _param_signature(params: Dict) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(params.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description='Export top distinct Optuna configs for 5-fold confirmation.')
    parser.add_argument('--storage', required=True, help='Optuna storage URL, e.g. sqlite:///optuna_results/study.db')
    parser.add_argument('--study_name', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--top_k', type=int, default=6)
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    candidates: List[Dict] = []
    seen = set()
    complete_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    complete_trials.sort(key=lambda trial: trial.value)

    for trial in complete_trials:
        signature = _param_signature(trial.params)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append({
            'source_trial_number': trial.number,
            'source_value': float(trial.value),
            'source_user_attrs': dict(trial.user_attrs),
            'params': dict(trial.params),
        })
        if len(candidates) >= args.top_k:
            break

    if not candidates:
        raise RuntimeError('No complete trials found; cannot prepare confirmation candidates.')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as output_file:
        yaml.safe_dump({'candidates': candidates}, output_file, sort_keys=False)
    print(f'Wrote {len(candidates)} candidates to {args.output}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run a declared baseline or frozen residual engineering evaluation."""
import argparse
import json
from dataclasses import asdict
import hashlib
from pathlib import Path
from sttw_control.env import RecoveryEnv,load_config
from sttw_control.evaluation import evaluate
from sttw_control.network import load_policy,make_policy_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','mjx'],default='cpu')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--policy',type=Path)
    args=p.parse_args()
    env=RecoveryEnv(load_config(args.config),backend=args.backend)
    identity=make_policy_identity(env.bundle.identity,asdict(env.config),env.config.observation.history_steps)
    policy=load_policy(args.policy,expected=identity) if args.policy else None
    if args.policy:
        identity={**identity,'checkpoint_sidecar_sha256':hashlib.sha256((args.policy/'identity.json').read_bytes()).hexdigest()}
    print(json.dumps(evaluate(env,args.output,seed=args.seed,policy=policy,policy_identity=identity if policy else None),indent=2))


if __name__=='__main__': main()

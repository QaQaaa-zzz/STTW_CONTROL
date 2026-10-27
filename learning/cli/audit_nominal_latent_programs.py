"""Read-only eligibility audit for matched-alpha nominal-program counterfactuals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sttw_control.nominal_latent_program import extract_nominal_program


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();review=json.loads(args.review.read_text())
    args.output.mkdir(exist_ok=False)
    source=Path(next(p for p in review['inputs'] if p.endswith('baselines_v1/analysis/review.json')))
    assert digest(source)==review['inputs'][str(source)]
    rows=json.loads(source.read_text())['records']
    index={(r['state_id'],r['force_N'],r['role']):r for r in rows}
    receipts=[]
    for row in review['records']:
        if row['force_N']!=0 or row['role'] not in ('future0','future1'):
            continue
        if float(row['role'][-1])!=row['original_user_alpha']:
            continue
        origin=index[row['state_id'],row['force_N'],row['role']]
        file=Path(origin['raw_file']);assert digest(file)==row['raw_sha256']
        with np.load(file) as z:
            program,receipt=extract_nominal_program(z)
        receipt.update(state_id=row['state_id'],original_user_alpha=row['original_user_alpha'],
                       raw_file=str(file),raw_sha256=digest(file))
        if program is not None:
            destination=args.output/(row['state_id']+'_program.npz')
            np.savez_compressed(destination,bare_latents=program,suffix_publication_s=np.arange(len(program))*.02)
            receipt.update(program_file=str(destination.resolve()),program_sha256=digest(destination))
        receipts.append(receipt)
    assert len(receipts)==4
    result=dict(question='Can original-user-alpha nominal programs be extracted without filling missing neural outputs?',
                source_review=str(args.review.resolve()),source_review_sha256=digest(args.review),
                records=receipts,eligible_states=sum(r['eligible'] for r in receipts),
                physics_ticks=0,training_steps=0,
                interpretation='Eligibility only; no replay implemented/executed, not evidence of feedback contribution. Availability selects alpha0; alpha1 remains untested.')
    (args.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

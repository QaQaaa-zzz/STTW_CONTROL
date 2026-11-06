"""Fixed-panel candidate selection, with nominal retention before task gains."""


def select_adapted_skill(rows):
    rows=sorted(rows,key=lambda row:row['update'])
    baseline=rows[0]
    if baseline['update']!=0:
        raise ValueError('Selection requires the original update0 panel')
    best=baseline;decisions=[]
    for row in rows:
        lost=[f'{case}:{metric}' for case,values in baseline['nominal'].items()
              for metric,value in values.items() if value and not row['nominal'][case][metric]]
        count=sum(v['success'] for v in row['hard'].values())
        previous=sum(v['success'] for v in best['hard'].values())
        common=[case for case,value in row['hard'].items() if value['success'] and best['hard'][case]['success']]
        difference=(sum(row['hard'][case]['time_s']-best['hard'][case]['time_s'] for case in common)/len(common)) if common else None
        better=not lost and (count>previous or (count==previous and difference is not None and difference < -1e-6))
        decisions.append(dict(update=row['update'],lost_nominal_conditions=lost,hard_successes=count,
                              compared_to_update=best['update'],shared_success_count=len(common),
                              shared_success_time_difference_s=difference,selected=better))
        if better:best=row
    return dict(selected_update=best['update'],checkpoint=best['checkpoint'],decisions=decisions,
                nominal_baseline_positive_conditions=sum(bool(v) for c in baseline['nominal'].values() for v in c.values()),
                scope='Best retention-qualified evaluated DEV candidate; zero baseline positives makes retention uninformative, not proof of preserved capability')

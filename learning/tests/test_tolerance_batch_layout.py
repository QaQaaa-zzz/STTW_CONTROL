from sttw_control.tolerance_batch_layout import contexts,lanes,case_name,FORCES,ONSETS

def test_full_declared_cross_product_and_unique_case_names():
 rows=lanes();assert len(rows)==88
 names=[case_name(c,t) for t in ONSETS for c in rows]
 assert len(names)==len(set(names))==264
 assert sum(c['data_role']=='calibration' for c in rows)==44
 assert sum(c['data_role']=='speed_transfer' for c in rows)==44

def test_branch_groups_share_context_and_start_from_zero_lane():
 rows=lanes()
 for i,c in enumerate(contexts()):
  group=rows[i*11:(i+1)*11]
  assert group[0]['force_N']==0
  assert tuple(x['force_N'] for x in group)==FORCES
  assert all(x['context_index']==i and all(x[k]==c[k] for k in c) for x in group)

def test_branch_indices_preserve_each_context_instead_of_global_lane_zero():
 from sttw_control.tolerance_batch_layout import prefix_indices
 indices=prefix_indices()
 assert len(indices)==88
 assert indices[:11]==[0]*11
 assert indices[11:22]==[11]*11
 assert indices[-11:]==[77]*11

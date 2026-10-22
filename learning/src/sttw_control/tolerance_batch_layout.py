"""Predeclared cross-state tolerance panel; pure layout, no simulation."""
FORCES=(0.,.5,-.5,2.,-2.,4.,-4.,6.,-6.,8.,-8.)
ONSETS=(1.,2.,4.)
def contexts():
 return [dict(route=route,speed=speed,alpha=alpha,data_role='calibration' if speed==2.3 else 'speed_transfer') for speed in [2.3,2.8] for route in ['left90_R2','right90_R2'] for alpha in [0,1]]
def lanes():
 return [dict(**c,context_index=i,force_N=f) for i,c in enumerate(contexts()) for f in FORCES]
def case_name(c,onset):
 return f"{c['route']}_v{c['speed']:g}_a{c['alpha']}_t{onset:g}_F{c['force_N']:+g}"
def prefix_indices():
 return [i*len(FORCES) for i in range(len(contexts())) for _ in FORCES]

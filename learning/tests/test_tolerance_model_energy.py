import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cli'))
import numpy as np
from audit_tolerance_model_energy import choose_threshold,score

def test_effort_budget_cannot_be_negative_even_for_ambiguous_zero_energy():
 assert choose_threshold([0.,0.],[True,False])>=0

def test_single_class_does_not_fake_identified_threshold():
 assert choose_threshold([1.,2.],[True,True]) is None

def test_separated_efforts_and_scaling():
 t=choose_threshold([1.,2.,4.,5.],[True,True,False,False]);assert 2<=t<4
 assert choose_threshold([10.,20.,40.,50.],[True,True,False,False])==10*t
 rows=[dict(energy_dynamic=x,actual_success=y) for x,y in [(1.,True),(5.,False)]]
 assert score(rows,t,'dynamic')['correct']==2

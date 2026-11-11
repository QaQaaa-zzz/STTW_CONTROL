"""Task constraints must change selection without overriding physical feasibility."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cli'))
from build_mechanism_forecast import task_selection


def candidate(alpha,path,speed,margin=0.4,physical=True):
    member=dict(physical_feasible=physical,path_peak_m=path,speed_rmse_m_s=speed)
    return dict(alpha=alpha,members=[member.copy(),member.copy()],physical_feasibility_vote=float(physical),mean_bottleneck_score=margin)


class TaskSelectionTests(unittest.TestCase):
    def test_path_and_speed_tasks_choose_different_modes(self):
        result=task_selection([candidate(0,.2,.15),candidate(1,.5,.08)])
        self.assertEqual(result['A']['alpha'],0)
        self.assertEqual(result['B']['alpha'],1)
    def test_physically_excluded_low_error_does_not_win(self):
        result=task_selection([candidate(0,.01,.01,physical=False),candidate(.5,.2,.12)])
        self.assertEqual(result['C']['alpha'],.5)
    def test_no_feasible_candidate_is_explicit(self):
        result=task_selection([candidate(0,.8,.3),candidate(.5,.7,.25)])
        self.assertEqual(result['A']['mode'],'least_error_physical_fallback')
        self.assertEqual(result['A']['feasible_vote'],0)
        self.assertEqual(result['A']['alpha'],.5)


if __name__=='__main__':unittest.main()

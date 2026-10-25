"""Process events and stage receipts govern the finite evaluation continuation."""
import os,select,subprocess,sys
import pytest


def test_pidfd_tracks_actual_child_exit_without_reading_training_status_in_a_loop():
    from sttw_control.survival_endpoint_sequence import process_identity,open_process_event
    child=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE)
    try:
        before=process_identity(child.pid)
        assert before['pid']==child.pid and before['start_ticks']>0
        fd=open_process_event(child.pid)
        try:
            poll=select.poll();poll.register(fd,select.POLLIN)
            assert not poll.poll(0)
            child.communicate(timeout=5)
            assert poll.poll(1000)
        finally:os.close(fd)
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_incomplete_or_failed_training_never_becomes_a_terminal_evaluation_source():
    from sttw_control.survival_endpoint_sequence import require_fixed_terminal
    with pytest.raises(ValueError):require_fixed_terminal(dict(phase='running',pid=7,completed_updates=63,declared_updates=64),7)
    with pytest.raises(ValueError):require_fixed_terminal(dict(phase='error',pid=7,completed_updates=64,declared_updates=64),7)
    with pytest.raises(ValueError):require_fixed_terminal(dict(phase='completed',pid=8,completed_updates=64,declared_updates=64),7)
    require_fixed_terminal(dict(phase='completed',pid=7,completed_updates=64,declared_updates=64),7)


def test_collection_counts_and_generated_figures_do_not_imply_visual_delivery_or_adoption():
    from sttw_control.survival_endpoint_sequence import require_stage_receipt
    with pytest.raises(ValueError):require_stage_receipt('collection',dict(phase='completed',completed_cases=487,endpoint_authenticated=True))
    require_stage_receipt('collection',dict(phase='completed',completed_cases=488,endpoint_authenticated=True))
    with pytest.raises(ValueError):require_stage_receipt('task_analysis',dict(phase='completed',completed_cases=487))
    require_stage_receipt('task_analysis',dict(phase='completed',completed_cases=488))
    require_stage_receipt('figures',dict(phase='completed',stage='generated_pending_visual_review'))

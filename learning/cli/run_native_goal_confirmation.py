"""Explicit prepare/shape/execute for one fixed native-permission80 panel."""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode=True

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','prefix-shape','prefix','shape','execute']);parser.add_argument('root',type=Path)
    parser.add_argument('--source',type=Path,default=Path('runs/native_zero_preference_v1'))
    parser.add_argument('--mode',choices=['h75_user_late_state192','user_preference120','tolerance_broaden120','tolerance_online40','h75_late_state_tolerance48','tolerance_online120','legacy','h75_channel_response','h75_channel_switching','h75_switching_continuation','h75_online_scalar','h75_online_confirmation100','h75_samestate_tolerance72','h75_samestate_bank144'],default='legacy')
    parser.add_argument('--protocol',type=Path,default=None)
    parser.add_argument('--charged-source',type=Path)
    args=parser.parse_args()
    if args.protocol is None:args.protocol=Path('learning/configs/h75_user_late_state192.json' if args.mode=='h75_user_late_state192' else 'learning/configs/h75_user_preference120.json' if args.mode=='user_preference120' else 'learning/configs/h75_tolerance_broaden120.json' if args.mode=='tolerance_broaden120' else 'learning/configs/h75_tolerance_online40.json' if args.mode=='tolerance_online40' else 'learning/configs/h75_late_state_tolerance48.json' if args.mode=='h75_late_state_tolerance48' else 'learning/configs/h75_tolerance_online120.json' if args.mode=='tolerance_online120' else 'learning/configs/h75_samestate_bank144.json' if args.mode=='h75_samestate_bank144' else 'learning/configs/h75_samestate_tolerance72.json' if args.mode=='h75_samestate_tolerance72' else 'learning/configs/h75_online_confirmation100.json' if args.mode=='h75_online_confirmation100' else 'learning/configs/h75_online_scalar.json' if args.mode=='h75_online_scalar' else 'learning/configs/h75_switching_continuation.json' if args.mode=='h75_switching_continuation' else 'learning/configs/h75_channel_switching.json' if args.mode=='h75_channel_switching' else 'learning/configs/h75_channel_response.json')
    file=Path(__file__).resolve().parents[1]/'src/sttw_control/native_goal_run.py'
    spec=importlib.util.spec_from_file_location('_sttw_native_goal_runner',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        if args.stage=='prepare':
            if args.mode=='h75_user_late_state192':module.prepare_late_state(args.source,args.root,args.protocol)
            elif args.mode=='user_preference120':module.prepare_user_preference(args.source,args.root,args.protocol)
            elif args.mode=='tolerance_broaden120':module.prepare_broaden(args.source,args.root,args.protocol)
            elif args.mode=='tolerance_online40':module.prepare_temporal_online(args.source,args.root,args.protocol)
            elif args.mode=='h75_late_state_tolerance48':module.prepare_late_state(args.source,args.root,args.protocol)
            elif args.mode=='tolerance_online120':module.prepare_tolerance(args.source,args.root,args.protocol)
            elif args.mode=='h75_samestate_bank144':module.prepare_bank144(args.source,args.root,args.protocol)
            elif args.mode=='h75_samestate_tolerance72':module.prepare_samestate(args.source,args.root,args.protocol)
            elif args.mode=='h75_online_confirmation100':module.prepare_confirmation(args.source,args.root,args.protocol)
            elif args.mode=='h75_online_scalar':module.prepare_online(args.source,args.root,args.protocol)
            elif args.mode=='h75_switching_continuation':module.prepare_continuation(args.source,args.root,args.protocol,charged_source=args.charged_source)
            elif args.mode=='h75_channel_switching':module.prepare_switching(args.source,args.root,args.protocol)
            elif args.mode=='h75_channel_response':module.prepare_response(args.source,args.root,args.protocol)
            else:module.prepare(args.source,args.root)
        elif args.stage in ('prefix','prefix-shape'):module.run_prefix(args.root,shape_only=args.stage=='prefix-shape')
        else:module.run(args.root,shape_only=args.stage=='shape')
    except BaseException:
        if args.stage=='execute':
            from engineer_reference_bias import notify
            notify(args.root,'STTW native goal比较报错','已停止并保存partial原始数据和计费，不自动重试或扩力。')
        raise
    else:
        if args.stage=='execute':
            from engineer_reference_bias import notify
            notify(args.root,'STTW native goal比较完成','冻结面板比较完成，无训练；随后分析原目标到达、实际触地、偏好代价及完整XY。')

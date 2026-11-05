"""Read the native Go2 training recipe without importing Isaac Gym into Lab.

The selected upstream classes contain literal configuration assignments. Only
configuration is parsed here; runner, storage, GAE, PPO and networks are imported
directly from the same checkout by the training entry point.
"""
import ast
from pathlib import Path


def spring_jump_recipe(checkout):
    source = Path(checkout) / 'legged_gym/envs/Go2_Flip/Go2_Spring_Jump/Go2_Spring_Jump_Config.py'
    classes = {node.name: node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef)}

    def section(parent, name):
        node = next(node for node in classes[parent].body
                    if isinstance(node, ast.ClassDef) and node.name == name)
        return {item.targets[0].id: ast.literal_eval(item.value)
                for item in node.body if isinstance(item, ast.Assign)}

    train = {name: section('Go2_Spring_Jump_PPO_Yu', name)
             for name in ('policy', 'algorithm', 'runner')}
    # Go2's 12-joint mirror mapping is not a two-wheel vehicle symmetry map.
    train['algorithm']['sym_loss'] = False
    for key in ('obs_permutation', 'act_permutation', 'frame_stack', 'sym_coef'):
        train['algorithm'].pop(key)
    return dict(source=str(source), train=train,
                domain_rand=section('Go2_Spring_Jump_Cfg_Yu', 'domain_rand'))

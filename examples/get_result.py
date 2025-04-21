import os
from typing import Sequence

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_PATH = os.path.join(PROJECT_ROOT, 'examples', 'runs')
RESULT_PATH = os.path.join(PROJECT_ROOT, 'result')

LOGTAGS = {
    'cost': 'Metrics/EpCost',
    'return': 'Metrics/EpRet',
}


def epoch_to_step(epoch):
    return (epoch + 1) * 4000


def extract_training_data(
    envs: Sequence[str],
    algs: Sequence[str],
    tags: Sequence[str],
):
    for env in envs:
        for alg in algs:
            env_alg_dir = os.path.join(LOG_PATH, f'{alg}-' + '{' + env + '}')
            for log_dir_name in os.listdir(env_alg_dir):
                log = pd.read_csv(os.path.join(env_alg_dir, log_dir_name, 'progress.csv'))
                step = epoch_to_step(log['Train/Epoch'])
                seed = str(int(log_dir_name.split('-')[1]))
                for tag in tags:
                    df = pd.DataFrame(
                        {
                            'step': step,
                            'value': smooth(log[LOGTAGS[tag]]),
                        }
                    )
                    result_dir = os.path.join(RESULT_PATH, env, tag)
                    os.makedirs(result_dir, exist_ok=True)
                    result_file_name = '_'.join([alg, seed]) + '.csv'
                    result_file = os.path.join(result_dir, result_file_name)
                    df.to_csv(result_file, index=False)


def smooth(x: pd.Series):
    # sliding window with length 2
    return pd.concat((x[:1], (x[:-1] + x[1:]) / 2))


envs = [
    'SafetyPointGoal1-v0',
    'SafetyPointPush1-v0',
    'SafetyPointCircle1-v0',
    'SafetyCarGoal1-v0',
    'SafetyCarPush1-v0',
    'SafetyCarCircle1-v0',
    'SafetyAntVelocity-v1',
    'SafetyHumanoidVelocity-v1',
    'SafetyPointButton1-v0',
    'SafetyCarButton1-v0',
    'SafetyHalfCheetahVelocity-v1',
    'SafetyHopperVelocity-v1',
    'SafetyWalker2dVelocity-v1',
    'SafetySwimmerVelocity-v1',
]

algs = [
    'CPO',
    'RCPO',
    'PPOLag',
    'FOCOPS',
    'CUP',
]

tags = [
    'cost',
    'return',
]

extract_training_data(envs, algs, tags)

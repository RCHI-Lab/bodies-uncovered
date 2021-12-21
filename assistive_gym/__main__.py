import argparse
from .env_viewer import viewer
from .envs.bm_config import BM_Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')

    bm_config = BM_Config()
    parser = bm_config.add_bm_args(parser)
    args = parser.parse_args()
    bm_config.change_bm_config(args)

    viewer(args.env)

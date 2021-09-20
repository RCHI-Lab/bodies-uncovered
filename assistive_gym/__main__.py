import argparse
from .env_viewer import viewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='BeddingManipulationSphere-v1', required=True,
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args, unknown = parser.parse_known_args()

    viewer(args.env)

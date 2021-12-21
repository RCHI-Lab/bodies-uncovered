import os, argparse, configparser

class BM_Config:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config.ini')
        self.configp = configparser.ConfigParser()
        self.configp.read(self.config_file)
        self.task = 'bedding_manipulation'
        self.reset_bm_config()

    def add_bm_args(self, parser):
        parser.add_argument('--target-limb-code', required=True,
                                help='Code for target limb to uncover, see human.py for a list of available target codes')
        parser.add_argument('--render-body-points', action='store_true', default=False,
                        help='Render points on the body. Points still exist even if not rendered')
        parser.add_argument('--bm-verbose', action='store_true', default=False,
                        help='More verbose prints')
        parser.add_argument('--take-images', action='store_true', default=False,
                        help='Enable taking images during rollout')
        parser.add_argument('--save-image-dir', default='./saved_images',
                        help='Directory to save images to')
        parser.add_argument('--fixed-human-pose', action='store_true', default=False,
                        help='Fixed human pose between rollouts')
        parser.add_argument('--vary-blanket-pose', action='store_true', default=False,
                        help='Introduce variation to the initial configuration of the blanket.')
        parser.add_argument('--vary-body-shape', action='store_true', default=False,
                        help='Introduce variation to human body shape.')
        parser.add_argument('--cmaes-data-collect', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
        return parser
    
    def reset_bm_config(self):
        self.configp.set(self.task, 'target_limb_code', 'None')
        self.configp.set(self.task, 'fixed_human_pose', 'False')
        self.configp.set(self.task, 'vary_blanket_pose', 'False')
        self.configp.set(self.task, 'vary_body_shape', 'False')
        self.configp.set(self.task, 'render_body_points', 'False')
        self.configp.set(self.task, 'bm_verbose', 'False')
        self.configp.set(self.task, 'take_images', 'False')
        self.configp.set(self.task, 'save_image_dir', '/saved_images')
        self.configp.set(self.task, 'cmaes_data_collect', 'False')

        with open(self.config_file, 'w') as configfile:
            self.configp.write(configfile)
    
    def change_bm_config(self, args):
        self.configp.set(self.task, 'target_limb_code', args.target_limb_code)
        self.configp.set(self.task, 'fixed_human_pose', str(args.fixed_human_pose))
        self.configp.set(self.task, 'vary_blanket_pose', str(args.vary_blanket_pose))
        self.configp.set(self.task, 'vary_body_shape', str(args.vary_body_shape))
        self.configp.set(self.task, 'render_body_points', str(args.render_body_points))
        self.configp.set(self.task, 'bm_verbose', str(args.bm_verbose))
        self.configp.set(self.task, 'take_images', str(args.take_images))
        self.configp.set(self.task, 'save_image_dir', args.save_image_dir)
        self.configp.set(self.task, 'cmaes_data_collect', str(args.cmaes_data_collect))

        with open(self.config_file, 'w') as configfile:
            self.configp.write(configfile)
from .bedding_manipulation import BeddingManipulationEnv
from .bedding_manipulation_action_parallel import BeddingManipulationParallelEnv
from .bedding_manipulation_action_series import BeddingManipulationSeriesEnv
from .agents import stretch, human
from .agents.stretch import Stretch
from .agents.human import Human

robot_arm = 'left'
human_controllable_joint_indices = []
class BeddingManipulationSphereEnv(BeddingManipulationEnv):
    def __init__(self):
        super(BeddingManipulationSphereEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=False))

class BMParallelSphereEnv(BeddingManipulationParallelEnv):
    def __init__(self):
        super(BMParallelSphereEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=False))

class BMSeriesSphereEnv(BeddingManipulationSeriesEnv):
    def __init__(self):
        super(BMSeriesSphereEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=False))

class BeddingManipulationStretchEnv(BeddingManipulationEnv):
    def __init__(self):
        super(BeddingManipulationStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
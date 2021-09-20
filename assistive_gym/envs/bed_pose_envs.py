from .bed_pose import BedPoseEnv
from .agents import pr2, stretch, human
from .agents.pr2 import PR2
from .agents.stretch import Stretch
from .agents.human import Human

robot_arm = 'left'
human_controllable_joint_indices = human.left_arm_joints
class BedPosePR2Env(BedPoseEnv):
    def __init__(self):
        super(BedPosePR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
class BedPosePR2MeshEnv(BedPoseEnv):
    def __init__(self):
        super(BedPosePR2MeshEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), use_mesh=True)

class BedPoseStretchEnv(BedPoseEnv):
    def __init__(self):
        super(BedPoseStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
class BedPoseStretchMeshEnv(BedPoseEnv):
    def __init__(self):
        super(BedPoseStretchMeshEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False), use_mesh=True)


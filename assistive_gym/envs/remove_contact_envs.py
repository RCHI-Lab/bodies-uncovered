from .remove_contact import RemoveContactEnv
from .agents import stretch, human
from .agents.stretch import Stretch
from .agents.human import Human

robot_arm = 'left'
human_controllable_joint_indices = human.left_arm_joints
class RemoveContactSphereEnv(RemoveContactEnv):
    def __init__(self):
        super(RemoveContactSphereEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=False), use_mesh=True)

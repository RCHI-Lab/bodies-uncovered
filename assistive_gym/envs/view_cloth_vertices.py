import os, time
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class ViewClothVerticesEnv(AssistiveEnv):
    def __init__(self):
        super(ViewClothVerticesEnv, self).__init__(robot=None, human=None, task='dressing', obs_robot_len=1, obs_human_len=1, frame_skip=5, time_step=0.0001)

    def step(self, action):
        self.take_step([])
        # self.take_step(np.zeros(7))
        return np.zeros(1), 0, False, {}

    def _get_obs(self, agent=None):
        return np.zeros(1)

    def reset(self):
        super(ViewClothVerticesEnv, self).reset()

        self.robot = None
        self.human = None
        self.build_assistive_env(None)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'sleeve_585v.obj'), scale=1.0, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'gown_696v.obj'), scale=1.0, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.75], flags=0)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=8, numSolverIterations=1, physicsClientId=self.id)
        # p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25, physicsClientId=self.id)

        self.take_step([])

        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        for i, v in enumerate(data[1]):
            p.addUserDebugText(text=str(i), textPosition=v, textColorRGB=[0, 0, 0], textSize=1, lifeTime=0, physicsClientId=self.id)

        p.setGravity(0, 0, 0, physicsClientId=self.id)

        ## self.init_env_variables()
        return self._get_obs()


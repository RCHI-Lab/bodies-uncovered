# Bodies Uncovered: Learning to Manipulate Real Blankets Around People via Physics Simulations

This code accompanies the submission:  
["Bodies Uncovered: Learning to Manipulate Real Blankets Around People via Physics Simulations"](https://arxiv.org/abs/2109.04930)  
Kavya Puthuveetil, Charles C. Kemp, and Zackory Erickson

This repository provides a version of [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) modified for this work, as well as additional task-specific functionality. **Although files for other assistive environments are included, ONLY the Bedding Manipulation enviornment is functional!**

## Citation
K. Puthuveetil, C. C. Kemp, and Z. Erickson, “Bodies Uncovered: Learning to Manipulate Real Blankets Around People via Physics Simulations,” 2021.
```
@misc{puthuveetil2021bodies,
      title={Bodies Uncovered: Learning to Manipulate Real Blankets Around People via Physics Simulations}, 
      author={Kavya Puthuveetil and Charles C. Kemp and Zackory Erickson},
      year={2021},
      eprint={2109.04930},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
## Install
For more details, check out the [installation guide for Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym/wiki/1.-Install). Just replace the lines that say `git clone https://github.com/Healthcare-Robotics/assistive-gym.git` and `cd assistive-gym` with:
```
git clone https://github.com/RCHI-Lab/bodies-uncovered.git
cd bodies-uncovered
```
Generating the actuated human model in the Bedding Manipulation environment relies on SMPL-X human mesh models. In order to use these models, you will need to create an account at https://smpl-x.is.tue.mpg.de/index.html and [download](https://smpl-x.is.tue.mpg.de/download.php) the mesh models. Once downloaded, extract the file and move the entire `smplx` directory to `bodies-uncovered/assistive_gym/envs/assets/smpl_models/`. Once complete, you should have several files with this format: `bodies-uncovered/assistive_gym/envs/assets/smpl_models/smplx/SMPLX_FEMALE.npz`. This step is REQUIRED to run the Bedding Manipulation enviornment!

## Basics
The Bedding Manipulation environment, built in [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym), can be visualized using the following command:
```
python3 -m assistive_gym --env "BeddingManipulationSphere-v1" --target-limb-code [VALUE] --[OPTIONAL FLAGS]
```

`--target-limb-code` is a REQUIRED flag that specifies a target limb on the body. The available code values and corresponding body parts are summarized in the table below:

| Target Limb Code | Body Part       | | Target Limb Code | Body Part      | | Target Limb Code | Body Part        |
| ---------------- | --------------- |-| ---------------- | -------------- |-| ---------------- | ---------------- |
| 0                | Right Hand      | | 6                | Left Hand      | | 12               | Both Lower Legs  |
| 1                | Right Lower Arm | | 7                | Left Forearm   | | 13               | Upper Body       |
| 2                | Right Arm       | | 8                | Left Arm       | | 14               | Lower Body       |
| 3                | Right Foot      | | 9                | Left Foot      | | 15               | Entire Body      |
| 4                | Right Lower Leg | | 10               | Left Lower Leg | | random           | Rand Select 1-15 |
| 5                | Right Leg       | | 11               | Left Leg       |

By default, the Bedding Manipulation environment is configured such that the human body and inital blanket position size is fixed, body pose is varied, and body points are not rendered. You can add the following optional flags to the command above to modify the default configuration or to explore additional functionality:

| Optional Flag          | Description                                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `--fixed-human-pose`   | Maintain the same human pose between rollouts (fixed seed).                                                     |
| `--vary-blanket-pose`  | Introduce variation to where the blanket is dropped from.                                                       |
| `--vary-body-shape`    | Introduce variation to the shape of the human within 160-185cm in height.                                       |
| `--render-body-points` | Render target/non-target points on the body. These points exist even when not rendered.                         |
| `--verbose`            | More verbose prints.                                                                                            |
| `--take-images`        | Take images at various critial points during a simulation rollout.                                              |
| `--save-image-dir`     | Specify directory to save images to if the `--take-images` is selected. Flag followed by path to the directory. |
| `--cmaes-data-collect` | Sets up the environment to collect data via CMA-ES.                                                             |





## Examples
### Test Out the Bedding Manipulation 
Let's test out the Bedding Manipulation environment! Here, the sphere manipulator will move to grasp and release points predefined in [`env_viewer.py`](https://github.com/Zackory/assistive-gym-fem/blob/33b88e14679935299042545b807b44e8dc2d43f5/assistive_gym/env_viewer.py#L28) to uncover a random target body part.
```
python3 -m assistive_gym --env "BeddingManipulationSphere-v1" --target-limb-code random --render-body-points --verbose
```
We can capture images while the bedding manipulation is performed to generate an image sequence similar to that seen in Figure 2 of the [paper](https://arxiv.org/abs/2109.04930). Images are captured from both top and side views.
```
python3 -m assistive_gym --env "BeddingManipulationSphere-v1" --target-limb-code random --render-body-points --take-images --save-image-dir ./bm_images/
```

### Training Bedding Manipulation Policies
**Train:** Let's train a proximal policy optimization (PPO) policy to uncover the right lower leg of a human whose pose is varied. We train this policy over 3000 rollouts to start:
```
python3 -m assistive_gym.learn --env "BeddingManipulationSphere-v1" --target-limb-code 4 --algo ppo --train --train-timesteps 3000 --save-dir ./trained_models/ppo_tl4_3k 
```
**Continue Training:** We want to train the policy for more timesteps to improve performance. Here, we resume training for a total of 5000 rollouts.
```
python3 -m assistive_gym.learn --env "BeddingManipulationSphere-v1" --target-limb-code 4 --algo ppo --train --train-timesteps 5000 --save-dir ./trained_models/ppo_tl4_5k  --load-policy-path ./trained_models/ppo_tl4_3k
```

### Render or Evaluate Trained Policies
We provide pretrained PPO policies (trained over 5000 rollouts) to uncover six different target body parts. We use the PPO policy trained to uncover the right lower leg with human pose variation in the examples below.

**Render**: We can render 10 rollouts of the policy with body points visualized using the `--render-body-points` flag and rewards printed (with additional information) using the `--verbose` flag.
```
python3 -m assistive_gym.learn --env "BeddingManipulationSphere-v1" --target-limb-code 4 --algo ppo --render --seed 0 --load-policy-path ./trained_models/ppo_tl4_5k  --render-episodes 10 --render-body-points
```
**Evaluate**: We evaluate this policy over 100 rollouts. The evaluation reports the mean rewards, standard deviation of the rewards, mean target points uncovered, mean non-target points uncovered, and the mean head points covered. Additional data from each rollout during the evaluation is saved to a pickle file.
```
python3 -m assistive_gym.learn --env "BeddingManipulationSphere-v1" --target-limb-code 4 --algo ppo --evaluate --eval-episodes 100 --seed 0 --verbose --load-policy-path ./trained_models/ppo_tl4_5k
```
**Generalize**: In order to assess how well the trained policies generalize to scenarios outside the training distribution, we evaluate the policies with changes in the simulation enviornment. For example, we evaluate the policy over 100 rollouts when the inital blanket configuration is randomized.
```
python3 -m assistive_gym.learn --env "BeddingManipulationSphere-v1" --target-limb-code 4 --algo ppo --evaluate --eval-episodes 100 --seed 0 --verbose --load-policy-path ./trained_models/ppo_tl4_5k --vary-blanket-pose
```

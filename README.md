# ManiSkill2-Real2Sim

This repository is forked from the [original ManiSkill2 repo](https://github.com/haosulab/ManiSkill2), with the following changes:
- Environment (`mani_skill2/envs`): We removed all environments irrelevant to real-to-sim evaluation, and we implemented real-to-sim evaluation environments under `maniskill2/envs/custom_scenes`. These custom environments act as an independent component of ManiSkill2, allowing for automatic integration into the original ManiSkill2 repository without necessitating any modifications.
- Robot agents: We added new robot implementations in `mani_skill2/agents/configs` and `mani_skill2/agents/robots`. The corresponding robot assets (URDFs) are in `mani_skill2/assets/descriptions`.
- Controllers: We modified `pd_joint_pos.py`, `pd_ee_pose.py`, and `__init__.py` under `mani_skill2/agents/controllers/`, along with `base_controller.py` and `utils.py` under `ManiSkill2_real2sim/mani_skill2/agents/`, to support more controller implementations. These scripts can be automatic integrated into the original ManiSkill2 repository.
- Object assets: We added custom objects in `data/custom` and custom scenes in `data/hab2_bench_assets` for real-to-sim evaluation purposes.
- Demo manual control script (`mani_skill2/examples/demo_manual_control_custom_envs.py`): The script is modified from `mani_skill2/examples/demo_manual_control.py` of the original ManiSkill2 repo to support custom real-to-sim environment creationg and visualization. See the script details for usage.


(Original ManiSkill2 docs: https://haosulab.github.io/ManiSkill2)


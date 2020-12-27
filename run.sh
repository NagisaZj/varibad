CUDA_VISIBLE_DEVICES=4 python main.py --env-type mujoco_cheetah_vel_varibad --seed 0

CUDA_VISIBLE_DEVICES=0 python main.py --env-type mujoco_reacher_varibad --seed 4

CUDA_VISIBLE_DEVICES=3 python main.py --env-type mujoco_walker_vel_varibad --seed 2

CUDA_VISIBLE_DEVICES=2 python main.py --env-type mujoco_walker_varibad --seed 9

CUDA_VISIBLE_DEVICES=7 python main.py --env-type mujoco_point_varibad --seed 0

CUDA_VISIBLE_DEVICES=2 python main.py --env-type mujoco_point_sub_varibad --seed 2

CUDA_VISIBLE_DEVICES=5 python main.py --env-type mujoco_metaworld_varibad --seed 1
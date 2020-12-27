import numpy as np


from gym.envs.mujoco.reacher import ReacherEnv as ReacherEnv_

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py

class ReacherGoalEnv(ReacherEnv_):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, max_episode_steps=200,**kwargs):
        self.goals = self.sample_tasks(n_tasks)
        self.goal_radius = 0.09
        self._goal = [0,0,0.01]
        self._max_episode_steps = max_episode_steps
        super(ReacherGoalEnv, self).__init__()
        self.reset_task(0)

    def get_all_task_idx(self):
        print(len(self.goals))
        return range(len(self.goals))

    def step(self, action):
        #print(action,self._goal)
        action = np.clip(action,-1,1)
        tmp_finger = self.get_body_com("fingertip")
        vec = self.get_body_com("fingertip") - self._goal

        reward_dist = - np.linalg.norm(vec)
        #print(vec,reward_dist)
        reward_ctrl = - np.square(action).sum()
        sparse_reward = self.sparsify_rewards(reward_dist)
        reward = reward_dist + reward_ctrl
        sparse_reward = sparse_reward + reward_ctrl
        reward = sparse_reward
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = False
        env_infos = dict(finger=tmp_finger.tolist(),reward_dist=reward_dist, reward_ctrl=reward_ctrl,sparse_reward=sparse_reward,task=self._goal)
        #print(env_infos['finger'])
        return ob, reward, done, env_infos

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        if r<-self.goal_radius:
            sparse_r = 0
        else:
            sparse_r = r + 0.2
        return sparse_r

    def sample_tasks(self, num_tasks):
        #np.random.seed(1337)
        radius = np.random.uniform(0.2,0.25, num_tasks)
        angles = np.random.uniform(0, np.pi, num_tasks)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        heights = np.ones((num_tasks,), dtype=np.float32) * 0.01
        #print(xs.shape,heights.shape)
        goals = np.stack([xs, ys,heights], axis=1)

        #goals = np.stack([goals, heights], axis=1)
        #np.random.shuffle(goals)
        #goals = goals.tolist()
        return goals

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()


    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return self._goal



    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
        # self.reset()
from Learner import ACLearner
import RL
from Spaces import Spaces

import os
import tensorflow as tf
import fire


#cudNN fix
os.environ["CUDA_VISIBLE_DEVICES"]="0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def run(learning_rate=1e-4,
        discount_factor=0.995,
        load_checkpoint=False,
        output_dir='checkpoints',   # folder where checkpoints are saved
        epsilon=0.01,               # must be the same value in SSBM_ENV, random move probability
        gae_lambda=1.0,
        alpha=8e-4,                 # Entropy bonus scale
        ep_length=20*10+1,
        batch_size=8,
        neg_scale=0.95,             # Scaling of negative rewards, breaks symmetry
        dist_scale=0.,              # Distance scale
        dmg_scale=0.008,             # Damage scale
        localhost=False,
        char='ganon',               # Must be the same in SSBM_ENV
        gpu=0,                      # If no gpu, set to -1
        ):

        params = {  'neg_scale': neg_scale,
                    'dist_scale': dist_scale,
                    'dmg_scale': dmg_scale
                 }

        spaces = Spaces(char)
        action_space = spaces.action_space
        state_shape = spaces.observation_space
        print(state_shape)

        net = RL.AC(state_shape=state_shape, action_dim=action_space.len, epsilon_greedy=epsilon,
                    lr=learning_rate, gamma=discount_factor, entropy_scale=alpha, gae_lambda=gae_lambda, gpu=gpu,
                    traj_length=ep_length, batch_size=batch_size, neg_scale=neg_scale)


        # Save and restore model
        checkpoint = tf.train.Checkpoint(net=net, actor=net.policy)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=output_dir, max_to_keep=10)

        if load_checkpoint:
            assert os.path.isdir(output_dir)
            path_ckpt = tf.train.latest_checkpoint(output_dir)
            status = checkpoint.restore(path_ckpt).expect_partial()
            status.assert_existing_objects_matched()
            print("Restored {}".format(path_ckpt))

            net.policy.epsilon.assign(epsilon)
            net.entropy_scale.assign(alpha)

        learner = ACLearner(net, checkpoint_manager, ep_length, params, batch_size, localhost)

        # Start accepting experience...
        learner.run()

if __name__ == '__main__':

    fire.Fire(run)

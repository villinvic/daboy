import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, GRU
import numpy as np
from tf_agents.utils import value_ops
import copy


class Distribution(object):
    def __init__(self, dim):
        self._dim = dim
        self._tiny = 1e-8

    @property
    def dim(self):
        raise self._dim

    def kl(self, old_dist, new_dist):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def likelihood_ratio(self, x, old_dist, new_dist):
        raise NotImplementedError

    def entropy(self, dist):
        raise NotImplementedError

    def log_likelihood_sym(self, x, dist):
        raise NotImplementedError

    def log_likelihood(self, xs, dist):
        raise NotImplementedError


class Categorical(Distribution):
    def kl(self, old_param, new_param):
        """
        Compute the KL divergence of two Categorical distribution as:
            p_1 * (\log p_1  - \log p_2)
        """
        old_prob, new_prob = old_param["prob"], new_param["prob"]
        return tf.reduce_sum(
            old_prob * (tf.math.log(old_prob + self._tiny) - tf.math.log(new_prob + self._tiny)))

    def likelihood_ratio(self, x, old_param, new_param):
        old_prob, new_prob = old_param["prob"], new_param["prob"]
        return (tf.reduce_sum(new_prob * x) + self._tiny) / (tf.reduce_sum(old_prob * x) + self._tiny)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
            \log \sum(p_i * x_i)

        :param x (tf.Tensor or np.ndarray): Values to compute log likelihood
        :param param (Dict): Dictionary that contains probabilities of outputs
        :return (tf.Tensor): Log probabilities
        """
        probs = param["prob"]
        assert probs.shape == x.shape, \
            "Different shape inputted. You might have forgotten to convert `x` to one-hot vector."
        return tf.math.log(tf.reduce_sum(probs * x, axis=1) + self._tiny)

    def sample(self, param, amount=1):
        probs = param["prob"]
        # NOTE: input to `tf.random.categorical` is log probabilities
        # For more details, see https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/random/categorical
        # [probs.shape[0], 1]
        return tf.random.categorical(tf.math.log(probs), amount)

    def entropy(self, param):
        probs = param["prob"]
        return -tf.reduce_sum(probs * tf.math.log(probs + self._tiny), axis=1)


class CategoricalActor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, epsilon,
                 name="CategoricalActor"):
        super().__init__(name=name)
        self.dist = Categorical(dim=action_dim)
        self.action_dim = action_dim
        self.state_ndim = len(state_shape)
        self.epsilon = tf.Variable(epsilon, name="Actor_epsilon", trainable=False, dtype=tf.float32)

        self.l1 = Dense(256, activation='relu', dtype='float32', name="L1")
        self.r = Reshape((1, 256))
        self.l2 = GRU(256, time_major=True, stateful=True, return_sequences=True)
        self.r2 = Reshape((256,))
        # self.l2 = Dense(256, activation='relu', dtype='float32', name="L2")
        self.prob = Dense(action_dim, name="prob", activation="softmax")

        # test
        self(tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def get_params(self):
        return self.get_weights()
        # return {
        #    "weights" : self.get_weights()
        # }

    def load_params(self, params):
        self.set_weights(params)

    def _compute_feature(self, states):
        features = self.l1(states)
        features = self.r(features)
        features = self.l2(features)
        return self.r2(features)

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """
        features = self._compute_feature(states)

        probs = self.prob(features) * (1.0 - self.epsilon) + self.epsilon / np.float32(self.action_dim)

        return {"prob": probs}

    def call(self, states):
        """
        Compute actions and log probability of the selected action

        :return action (tf.Tensors): Tensor of actions
        :return log_probs (tf.Tensor): Tensors of log probabilities of selected actions
        """
        param = self._compute_dist(states)

        action = tf.squeeze(self.dist.sample(param), axis=1)  # (size,)

        log_prob = self.dist.log_likelihood(
            tf.one_hot(indices=action, depth=self.action_dim), param)

        return action, log_prob, param

    def get_probs(self, states):
        return self._compute_dist(states)["prob"]

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)

    def compute_log_probs(self, states, actions):
        """Compute log probabilities of inputted actions

        :param states (tf.Tensor): Tensors of inputs to NN
        :param actions (tf.Tensor): Tensors of NOT one-hot vector.
            They will be converted to one-hot vector inside this function.
        """
        param = self._compute_dist(states)
        actions = tf.one_hot(
            indices=tf.squeeze(actions),
            depth=self.action_dim)
        param["prob"] = tf.cond(
            tf.math.greater(tf.rank(actions), tf.rank(param["prob"])),
            lambda: tf.expand_dims(param["prob"], axis=0),
            lambda: param["prob"])
        actions = tf.cond(
            tf.math.greater(tf.rank(param["prob"]), tf.rank(actions)),
            lambda: tf.expand_dims(actions, axis=0),
            lambda: actions)
        log_prob = self.dist.log_likelihood(actions, param)
        return log_prob

    def get_action(self, state):
        # assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = state[np.newaxis].astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(state)

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state):
        param = self._compute_dist(state)
        return tf.squeeze(self.dist.sample(param), axis=1)


class Q(tf.keras.Model):
    """
    Compared with original (continuous) version of SAC, the output of Q-function moves
        from Q: S x A -> R
        to   Q: S -> R^|A|
    """

    def __init__(self, state_shape, action_dim, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1", activation='relu')
        self.l2 = Dense(256, name="L2", activation='relu')
        self.l3 = Dense(action_dim, name="L3", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)

    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return values


class V(tf.keras.Model):
    """
    Compared with original (continuous) version of SAC, the output of Q-function moves
        from Q: S x A -> R
        to   Q: S -> R^|A|
    """

    def __init__(self, state_shape, name='vf'):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1", activation='relu')
        self.r = Reshape((1, 256))
        self.l2 = GRU(256, time_major=True, stateful=True, return_sequences=True)
        self.r2 = Reshape((256,))
        # self.l2 = Dense(256, activation='relu', dtype='float32', name="L2")
        self.l3 = Dense(1, name="L3", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)

    def call(self, states):
        features = self.l1(states)
        features = self.r(features)
        features = self.l2(features)
        features = self.r2(features)
        value = self.l3(features)

        return value


class AC(tf.keras.Model):
    def __init__(self, state_shape, action_dim, epsilon_greedy, lr, gamma, entropy_scale, gae_lambda, gpu=0, traj_length=1,
                 name='AC'):
        super().__init__(name=name)
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = tf.expand_dims(
            tf.Variable(np.array([gae_lambda for _ in range(traj_length - 1)]), dtype=tf.float32, trainable=False), axis=1)
        self.epsilon = tf.Variable(0.001, dtype=tf.float32, trainable=False)

        self.V = V(state_shape)
        self.policy = CategoricalActor(state_shape, action_dim, epsilon_greedy)
        self.V_optim = tf.keras.optimizers.Adam(learning_rate=lr * 0.5, beta_1=0.9, beta_2=0.98, epsilon=1e-8)
        self.p_optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-8)
        self.ent_optim = tf.keras.optimizers.Adam(learning_rate=lr * 0.1, beta_1=0.9, beta_2=0.98, epsilon=1e-8)

        self.step = tf.Variable(0, dtype=tf.int32)
        self.batch_size = tf.Variable(traj_length - 1, dtype=tf.int32, trainable=False)

        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

        self.gae_values = tf.Variable(np.zeros((traj_length - 1,)), trainable=False, dtype=tf.float32)

        self.entropy_scale = tf.Variable(entropy_scale, dtype=tf.float32, trainable=True,
                                         constraint=tf.keras.activations.relu)
        self.max_entropy = tf.Variable(3.6, trainable=False, dtype=tf.float32)
        self.target_entropy = tf.Variable(2.0, trainable=False, dtype=tf.float32)

    def train(self, states, actions, rewards, dones):
        # do some stuff with arrays

        # print(states, actions, rewards, dones)

        policy_loss, std_error, mean_entropy, min_entropy, max_entropy, rew_ent_diff, min_logp, max_logp \
            = self._train(states, actions, rewards, dones)

        tf.summary.scalar(name=self.name + "/policy_loss", data=policy_loss)
        tf.summary.scalar(name=self.name + "/std", data=std_error)
        tf.summary.scalar(name=self.name + "/min_entropy", data=min_entropy)
        tf.summary.scalar(name=self.name + "/max_entropy", data=max_entropy)
        tf.summary.scalar(name=self.name + "/rew_ent_diff", data=rew_ent_diff)
        tf.summary.scalar(name=self.name + "/ent_scale", data=self.entropy_scale)
        tf.summary.scalar(name="logp/min_logp", data=min_logp)
        tf.summary.scalar(name="logp/max_logp", data=max_logp)

        return mean_entropy

    @tf.function
    def _train(self, states, actions, rewards, dones):
        with tf.device(self.device):
            not_dones = tf.expand_dims(1. - tf.cast(dones, dtype=tf.float32), axis=1)
            # rewards = tf.expand_dims(rewards, axis=1)
            actions = tf.cast(actions, dtype=tf.int32)

            with tf.GradientTape() as tape:
                v_all = self.V(states)
                v = v_all[:-1]
                last_v = v_all[-1]
                self.V.l2.reset_states()
                targets = self.compute_gae(v, rewards, last_v)
                advantage = tf.stop_gradient(targets) - v
                # v_next = v_all[1:]
                # v = v_all[:-1]
                # advantage = rewards + self.gamma * v_next * not_dones - v
                # advantage = self.compute_gae( deltas, rewards, v[-1])
                v_loss = tf.reduce_mean(tf.square(advantage))

            v_grad = tape.gradient(v_loss, self.V.trainable_variables)
            self.V_optim.apply_gradients(zip(v_grad, self.V.trainable_variables))

            with tf.GradientTape() as tape2:
                p = self.policy.get_probs(states[:-1])
                self.policy.l2.reset_states()
                p_log = tf.math.log(p + 1e-8)

                ent = - tf.reduce_sum(tf.multiply(p_log, p), -1)
                # dist = tf.stop_gradient((self.max_entropy - ent) / self.max_entropy)
                range = tf.expand_dims(tf.range(self.batch_size), axis=1)
                indices = tf.concat(values=[range, actions], axis=1)

                policy_loss = - tf.reduce_mean(
                    tf.expand_dims(tf.gather_nd(p_log, indices), axis=1) * advantage + self.entropy_scale * ent)

            p_grad = tape2.gradient(policy_loss, self.policy.trainable_variables)
            self.p_optim.apply_gradients(zip(p_grad, self.policy.trainable_variables))

            # with tf.GradientTape() as tape3:
            #    ent_scale_loss = -tf.reduce_mean( self.entropy_scale * ( self.target_entropy - ent) / self.target_entropy)
            #
            # ent_grad = tape3.gradient(ent_scale_loss, [self.entropy_scale])
            # self.ent_optim.apply_gradients(zip(ent_grad, [self.entropy_scale]))

            self.step.assign_add(1)
            mean_entropy = tf.reduce_mean(ent)
            min_entropy = tf.reduce_min(ent)
            max_entropy = tf.reduce_max(ent)
            diff = tf.reduce_mean(tf.divide(tf.abs(tf.expand_dims(tf.gather_nd(p_log, indices), axis=1) * advantage),
                                            self.entropy_scale * ent))

            return policy_loss, v_loss, mean_entropy, min_entropy, max_entropy, diff, tf.reduce_min(
                p_log), tf.reduce_max(p_log)

    def compute_gae(self, v, rewards, last_v):
        def bellman(future, present):
            val, r, l = present
            return (1. - l) * val + l * (r + self.gamma * future)

        reversed_sequence = [tf.reverse(t, [0]) for t in [v, rewards, self.gae_lambda]]
        returns = tf.scan(bellman, reversed_sequence, last_v)
        returns = tf.reverse(returns, [0])
        return returns


class RERPI(tf.keras.Model):

    def __init__(self, state_shape, action_dim, epsilon_greedy, lr, gamma, nu, gpu=0, target_update_freq=250,
                 name='RERPI'):
        super().__init__(name=name)
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.nu = tf.Variable(nu, dtype=tf.float32)
        self.lmultiplier = tf.Variable(10.0, dtype=tf.float32)
        self.lmultiplier_optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-8)
        self.epsilon = tf.Variable(0.001, dtype=tf.float32, trainable=False)

        self.Q = Q(state_shape, action_dim)
        self.Q_targ = Q(state_shape, action_dim)
        self.policy = CategoricalActor(state_shape, action_dim, epsilon_greedy)
        self.policy_theta = CategoricalActor(state_shape, action_dim, epsilon_greedy)
        self.Q_optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-8)
        self.p_optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-8)

        self.target_update_frq = target_update_freq
        self.step = tf.Variable(0, dtype=tf.int32)

        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    def exp_ranking(self, rank):
        return tf.math.pow((self.nu + self.action_dim) / (tf.cast(rank, tf.float32) + 1.0), 3.0)

    def train(self, states, actions, rewards, dones):
        # do some stuff with arrays

        # print(states, actions, rewards, dones)
        self.policy_theta.set_weights(self.policy.get_weights())
        policy_loss, std_error, mean_entropy, min_entropy, max_entropy = self._train(states, actions, rewards, dones)

        if self.step % self.target_update_frq == 0:
            self.Q_targ.set_weights(self.Q.get_weights())
            # tf.print(self.lmultiplier)
        tf.summary.scalar(name=self.name + "/LR", data=policy_loss)
        tf.summary.scalar(name=self.name + "/L_mult", data=self.lmultiplier)
        tf.summary.scalar(name=self.name + "/std", data=std_error)
        tf.summary.scalar(name=self.name + "/mean_entropy", data=mean_entropy)
        tf.summary.scalar(name=self.name + "/min_entropy", data=min_entropy)
        tf.summary.scalar(name=self.name + "/max_entropy", data=max_entropy)

        return policy_loss, std_error, mean_entropy

    @tf.function
    def policy_evaluation(self, states, indices, rewards, not_dones):
        with tf.GradientTape() as g:
            # policyEvaluation
            target_q = self.Q_targ(states[1:])
            q = self.Q(states[:-1])

            std_error = tf.reduce_mean(tf.square(rewards + not_dones * target_q * self.gamma - q))

        q_grad = g.gradient(std_error, self.Q.trainable_variables)
        self.Q_optim.apply_gradients(zip(q_grad, self.Q.trainable_variables))

        return std_error

    @tf.function
    def policy_improvement(self, states, q):
        ranked = tf.nn.top_k(q, k=self.action_dim)
        indices = tf.cast(ranked.indices, dtype=tf.float32)
        coef = self.exp_ranking(indices)
        sum_all = tf.reduce_sum(coef[0])
        weights = tf.divide(coef, sum_all)
        # Eq 4
        # with tf.GradientTape(watch_accessed_variables=False) as g:
        #    g.watch(self.lmultiplier)
        #    lm_loss = self._compute_Lagrange(weights, states)

        # lmultiplier_grad = g.gradient(lm_loss, self.lmultiplier)
        # self.lmultiplier_optim.apply_gradients([(lmultiplier_grad, self.lmultiplier)])

        with tf.GradientTape(watch_accessed_variables=False) as gg:
            gg.watch(self.policy.trainable_variables)
            parametric_loss = -self._compute_Lagrange(weights, states)

        parametric_grad = gg.gradient(parametric_loss, self.policy.trainable_variables)

        self.p_optim.apply_gradients(zip(parametric_grad, self.policy.trainable_variables))

        return parametric_loss

    @tf.function
    def _compute_Lagrange(self, weights, states):
        probs = self.policy.get_probs(states[:-1])
        #   theta_probs = self.policy_theta.get_probs(states[:-1])
        logp = tf.math.log(probs)
        t1 = tf.multiply(weights, logp)
        tt1 = tf.reduce_sum(t1)
        # tf.print(t1, tt1)
        # t2 = self.lmultiplier * (self.epsilon - tf.reduce_sum( tf.losses.KLD(theta_probs, probs) / self.action_dim))
        return tt1  # + t2

    @tf.function
    def _train(self, states, actions, rewards, dones):
        with tf.device(self.device):
            batch_size = actions.shape[0]
            not_dones = tf.expand_dims(1. - tf.cast(dones, dtype=tf.float32), axis=1)
            rewards = tf.expand_dims(rewards, axis=1)
            actions = tf.expand_dims(tf.cast(actions, dtype=tf.int32), axis=1)
            range = tf.expand_dims(tf.range(batch_size), axis=1)
            indices = tf.concat(
                values=[range, actions], axis=1)

            std_error = self.policy_evaluation(states, indices, rewards, not_dones)

            # policyImprovement
            """         
            π(k+1) = argminπθEµπ(s)[KL q(a|s)|| πθ(a|s)]
            """
            # weighting actions, sampling all ?
            # sample N states, K actions
            qq = self.Q(states[:-1])

            param_loss = self.policy_improvement(states, qq)

            self.step.assign_add(1)

            probs = self.policy.get_probs(states[:-1])
            logp = tf.math.log(probs)
            ent = -tf.multiply(logp, probs)
            mean_entropy = tf.reduce_mean(ent)
            min_entropy = tf.reduce_min(ent)
            max_entropy = tf.reduce_max(ent)
            # tf.print(self.epsilon - tf.reduce_sum( tf.losses.KLD(probs, self.policy_theta.get_probs(states[:-1])) / self.action_dim))

            return -param_loss, std_error, mean_entropy, min_entropy, max_entropy

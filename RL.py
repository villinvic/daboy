import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU
import numpy as np
from tensorflow.keras.activations import relu


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
        #tf.print(probs, tf.math.log(probs), tf.random.categorical(tf.math.log(probs), amount), summarize=-1)
        return tf.cast(tf.map_fn( lambda p: tf.cast(tf.random.categorical(tf.math.log(p), amount), tf.float32),  probs), tf.int64)

    def entropy(self, param):
        probs = param["prob"]
        return -tf.reduce_sum(probs * tf.math.log(probs + self._tiny), axis=1)


class CategoricalActor(tf.keras.Model):
    '''
    Actor model class
    '''

    def __init__(self, state_shape, batch_size, traj_length, action_dim, epsilon,
                 name="CategoricalActor"):
        super().__init__(name=name)
        self.dist = Categorical(dim=action_dim)
        self.action_dim = action_dim
        self.state_ndim = len(state_shape)
        self.epsilon = tf.Variable(epsilon, name="Actor_epsilon", trainable=False, dtype=tf.float32)

        # self.l1 = Dense(128, activation='elu', dtype='float32', name="critic_L1")
        self.l1 = GRU(512, time_major=False, dtype='float32', stateful=True, return_sequences=True, return_state=True)
        self.l2 = Dense(128, activation='elu', dtype='float32', name="L2")
        # self.l3 = Dense(128, activation='elu', dtype='float32', name="L3")
        # self.l2 = Dense(128, activation='relu', dtype='float32', name="L2")
        self.prob = Dense(action_dim, dtype='float32', name="prob", activation="softmax")

        # test
        #self(tf.constant(
        #    np.zeros(shape=(None, None,) + state_shape, dtype=np.float32))) # Batch, traj
            
        tf.print((batch_size, traj_length,) + state_shape)

    def get_params(self):
        return {
           "weights" : self.get_weights()
        }

    def load_params(self, params):
        try:
            self.set_weights(params)
        except Exception: # Sometimes fail at beginning of training, tensor not fully initialized ?
            pass

    def _compute_feature(self, states):
        features, c_state = self.l1(states)
        features = self.l2(features)
        return features, c_state

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """
        features, c_state = self._compute_feature(states)

        probs = self.prob(features) * (1.0 - self.epsilon) + self.epsilon / np.float32(self.action_dim)

        return {"prob": probs,
                "recurrent_state": c_state}

    def get_action(self, state):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = state[np.newaxis][np.newaxis].astype(
            np.float32) if is_single_state else state
        action, recurrent_state = self._get_action_body(tf.constant(state))

        return (action.numpy()[0][0], recurrent_state.numpy()) if is_single_state else (action, recurrent_state.numpy())

    @tf.function
    def _get_action_body(self, state):
        param = self._compute_dist(state)
        action = tf.squeeze(self.dist.sample(param), axis=1)
        return action, param['recurrent_state']

    def call(self, states):
        """
        Compute actions and log probability of the selected action

        :return action (tf.Tensors): Tensor of actions
        :return log_probs (tf.Tensor): Tensors of log probabilities of selected actions
        """
        param = self._compute_dist(states)

        action = tf.squeeze(self.dist.sample(param), axis=2)  # (size,)

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


class ActionStateProbs(tf.keras.Model):
    '''
    Unconditional model representing probability distribution for each action state.
    '''

    def __init__(self, name='qf'):
        super().__init__(name=name)
        self.one = tf.constant([[1.0]], dtype=tf.float32)
        self.probs = Dense(384, name="as_probs", activation='softmax')

    def call(self, x):
        return self.probs(self.one)


class V(tf.keras.Model):
    """
    Value model function
    """

    def __init__(self, state_shape, batch_size, traj_length, name='vf'):
        super().__init__(name=name)

        #self.l1 = Dense(128, activation='elu', dtype='float32', name="v_L1")
        self.l1 = GRU(512, time_major=False, dtype='float32', stateful=True, return_sequences=True)
        self.l2 = Dense(128, activation='elu', dtype='float32', name="L2")
        # self.l3 = Dense(128, activation='elu', dtype='float32', name="L3")
        self.v = Dense(1, activation='linear', dtype='float32', name="v")

    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        # features = self.l3(features)
        value = self.v(features)
        return value
        
        
class Predictor(tf.keras.Model):
    '''
    Unused
    '''
    def __init__(self, state_shape, batch_size, traj_length, name='pred'):
        super().__init__(name=name)

        self.l1 = Dense(256, activation='elu', dtype='float32', name="L1")
        self.l2 = GRU(512, time_major=False, stateful=True, return_sequences=True)
        
        self.l3 = Dense(state_shape[0], activation='linear', dtype='float32', name="L3")

    def call(self, states):
        features = self.l2(states)
        features = self.l1(features)
        states_ = self.l3(features)

        return states_


class AC(tf.keras.Model):
    def __init__(self, state_shape, action_dim, epsilon_greedy, lr, gamma, entropy_scale, gae_lambda, gpu=0, traj_length=1, batch_size=1, neg_scale=1.0,
                 train_predict=False, name='AC'):
        super().__init__(name=name)
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.neg_scale = neg_scale
        self.as_scale = 0.003
        self.train_predictor = train_predict
        self.gae_lambda = tf.Variable(gae_lambda, dtype=tf.float32, trainable=False)
        self.epsilon = tf.Variable(0.001, dtype=tf.float32, trainable=False)

        self.V = V(state_shape, batch_size, traj_length)
        self.policy = CategoricalActor(state_shape, batch_size, traj_length-1, action_dim, epsilon_greedy)
        self.as_probs = ActionStateProbs()
        self.p_optim = tf.keras.optimizers.SGD(learning_rate=1.0)
        self.optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-8, clipvalue=3.3e-3)
        self.step = tf.Variable(0, dtype=tf.int32)
        self.traj_length = tf.Variable(traj_length - 1, dtype=tf.int32, trainable=False)

        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

        self.entropy_scale = tf.Variable(entropy_scale, dtype=tf.float32, trainable=True,
                                         constraint=tf.keras.activations.relu)
        
        self.range_ = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(self.traj_length), axis=0), [self.batch_size , 1]), axis=2)
        self.pattern = tf.expand_dims([tf.fill((self.traj_length,), i) for i in range(self.batch_size)], axis=2)

    def train(self, states, actions, rewards, r_states):
        # do some stuff with arrays
        # print(states, actions, rewards, dones)
        if tf.reduce_any(tf.math.is_nan(states)):
                print(list(states))
                states = tf.where(tf.math.is_nan(states), tf.zeros_like(states), states)

        # Set both networks with corresponding initial recurrent state
        self.policy.l1.reset_states(states=r_states)
        self.V.l1.reset_states(states=r_states)

        v_loss, total_loss, mean_entropy, min_entropy, max_entropy, min_logp, max_logp, grad_norm \
            = self._train(states, actions, rewards)

        tf.summary.scalar(name=self.name + "/v_loss", data=v_loss)
        tf.summary.scalar(name=self.name + "/as_ent", data=total_loss)
        tf.summary.scalar(name=self.name + "/min_entropy", data=min_entropy)
        tf.summary.scalar(name=self.name + "/max_entropy", data=max_entropy)
        tf.summary.scalar(name=self.name + "/mean_entropy", data=mean_entropy)
        tf.summary.scalar(name=self.name + "/ent_scale", data=self.entropy_scale)
        tf.summary.scalar(name="logp/min_logp", data=min_logp)
        tf.summary.scalar(name="logp/max_logp", data=max_logp)
        tf.summary.scalar(name=self.name + "/grad_norm", data=grad_norm)
        tf.summary.scalar(name="misc/distance", data=tf.reduce_mean(states[:, :, -1]))
        

    @tf.function
    def _train(self, states, actions, rewards):
        '''
        Main training function
        '''

        with tf.device(self.device):
            
            actions = tf.cast(actions, dtype=tf.int32)
            with tf.GradientTape() as tape:
                # Update the action_state probability distribution

                as_probs = self.as_probs(None)[0]
                
                action_states = tf.cast(tf.argmax(states[:, :-1, 401 + 1: 401 + 1 + (383+1)], axis=2),  dtype=tf.int32)

                taken_as = tf.gather_nd(as_probs,tf.expand_dims(action_states, axis=2), batch_dims=0)
                NLL = -tf.math.log(taken_as + 1e-8)
                loss = tf.reduce_mean(NLL)
                
            grad = tape.gradient(loss, self.as_probs.trainable_variables)
            self.p_optim.apply_gradients(zip(grad, self.as_probs.trainable_variables))

            # Reward rare action_states with their negative log likelihood
            rewards += tf.clip_by_value(NLL-tf.math.log(384.0), clip_value_min=0.0, clip_value_max=5.0) * self.as_scale
            as_ent = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.math.log(as_probs), as_probs), -1))

            with tf.GradientTape() as tape:
                # Optimize the actor and critic

                v_all = self.V(states)
                v = v_all[:, :-1, 0]
                last_v = v_all[:, -1, 0]
                targets = self.compute_gae(v, rewards, last_v)
                advantage = tf.stop_gradient(targets) - v
                v_loss = tf.reduce_mean(tf.square(advantage))
                if v_loss > 0.5:
                    tf.print(advantage, summarize=-1)
                
                p = self.policy.get_probs(states[:, :-1])
                p_log = tf.math.log(p + 1e-8)
           
                ent = - tf.reduce_sum(tf.multiply(p_log, p), -1)
                indices = tf.concat(values=[ self.pattern, self.range_, tf.expand_dims(actions, axis=2)], axis=2)

                taken_p_log = tf.gather_nd(p_log, indices, batch_dims=0)

                p_loss = - tf.reduce_mean(
                    taken_p_log * tf.stop_gradient(advantage) + self.entropy_scale * ent)
                    
                total_loss = 0.5 * v_loss + p_loss
            
            grad = tape.gradient(total_loss, self.policy.trainable_variables + self.V.trainable_variables)

            # x is used to track the gradient size
            x = 0.0
            c = 0.0
            for gg in grad:
                c += 1.0
                x += tf.reduce_mean(tf.abs(gg))
            x /= c

            self.optim.apply_gradients(zip(grad, self.policy.trainable_variables + self.V.trainable_variables))

            self.step.assign_add(1)
            mean_entropy = tf.reduce_mean(ent)
            min_entropy = tf.reduce_min(ent)
            max_entropy = tf.reduce_max(ent)
            
            return v_loss, as_ent, mean_entropy, min_entropy, max_entropy, tf.reduce_min(
                p_log), tf.reduce_max(p_log), x

    def compute_gae(self, v, rewards, last_v):
        v = tf.transpose(v)
        rewards= tf.transpose(rewards)
        reversed_sequence = [tf.reverse(t, [0]) for t in [v, rewards]]
        
        def bellman(future, present):
            val, r = present
            # m = tf.cast(r > -0.9, tf.float32)
            clipped_r = tf.clip_by_value(r, clip_value_min=-2.0, clip_value_max=2.0)
            return (1. - self.gae_lambda) * val + self.gae_lambda * (clipped_r + (1.0-self.neg_scale)*relu(-clipped_r) + self.gamma * future)
        
        returns = tf.scan(bellman, reversed_sequence, last_v)
        returns = tf.reverse(returns, [0])
        returns = tf.transpose(returns)
        return returns
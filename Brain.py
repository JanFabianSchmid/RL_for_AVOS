import tensorflow as tf
import numpy as np
import random

DQN = {
    "FRAME_PER_ACTION": 1,
    "GAMMA": 1.0,              # decay rate of past observations
    "EXPLORE": 100000.,         # frames over which to anneal epsilon
    "FINAL_EPSILON": 0.001,     # final value of epsilon
    "INITIAL_EPSILON": 0.1,    # starting value of epsilon
    "REPLAY_MEMORY": 15000,     # number of previous experiences to remember
    "BATCH_SIZE": 32,           # size of minibatch
    "UPDATE_TIME": 200,         # number of training episodes until target network is updated
    "SAVE_TIME": 5000,         # number of training episodes until session is saved
    "histogram_log_time": 500,  # number of training episodes until histograms are written out
    "TRAIN_EVERY": 10,          # how often is the network trained
    "MAX_RANGE_OF_RNN": 0,      # maximum number of steps the RNN is trained with
    "RNN_LAYERS": 0,
    "learning_rate": 0.00001,
    "learning_rate_classifier": 0.00001,
    "episodic": False,
    "DQN": True,
    "proposal_classifier": False,
    "scene_classifier": False,
    "proposal_classifier_as_auxiliary": False,
    "element_wise_keep_prob": 0.75,
    "input_keep_prob": 0.85,
    "element_wise_keep_prob_for_input_drop": 0.9,
    "pos_ratio": 0.5,
    "image_noise": 0.05,
    "object_detector_noise": 0.1,
    "unified_aggregate": True,
    "log_histogram": False,
}

DRQN = {
    "FRAME_PER_ACTION": 1,
    "GAMMA": 1.0,               # decay rate of past observations
    "EXPLORE": 100000.,          # frames over which to anneal epsilon
    "FINAL_EPSILON": 0.001,      # final value of epsilon
    "INITIAL_EPSILON": 0.1,      # starting value of epsilon
    "REPLAY_MEMORY": 7500,       # number of previous episodes to remember
    "BATCH_SIZE": 16,            # size of minibatch
    "UPDATE_TIME": 200,          # number of training episodes until target network is updated
    "SAVE_TIME": 5000,            # number of training episodes until session is saved
    "histogram_log_time": 1000,  # number of training episodes until histograms are written out
    "TRAIN_EVERY": 25,           # how often is the network trained
    "MAX_RANGE_OF_RNN": 30,      # maximum number of steps the RNN is trained with
    "RNN_LAYERS": 3,
    "learning_rate": 0.00001,
    "learning_rate_classifier": 0.00001,
    "episodic": True,
    "DQN": True,
    "proposal_classifier": False,
    "scene_classifier": False,
    "proposal_classifier_as_auxiliary": False,
    "element_wise_keep_prob": 0.75,
    "input_keep_prob": 0.85,
    "element_wise_keep_prob_for_input_drop": 0.9,
    "pos_ratio": 0.5,
    "image_noise": 0.05,
    "object_detector_noise": 0.1,
    "unified_aggregate": True,
    "log_histogram": False,
}

# DRASCOS-b
DRASCOS = DRQN.copy()
DRASCOS["proposal_classifier"] = True
DRASCOS["scene_classifier"] = True

# DRASCOS-a
DRQN_with_auxiliary = DRQN.copy()
DRQN_with_auxiliary["proposal_classifier"] = True
DRQN_with_auxiliary["scene_classifier"] = True
DRQN_with_auxiliary["proposal_classifier_as_auxiliary"] = True

DRQN_with_scene= DRQN.copy()
DRQN_with_scene["scene_classifier"] = True

DRQN_with_prop = DRQN.copy()
DRQN_with_prop["proposal_classifier"] = True
DRQN_with_prop["proposal_classifier_as_auxiliary"] = True

DRQN_with_prop_small = DRQN.copy()
DRQN_with_prop_small["proposal_classifier"] = True
DRQN_with_prop_small["proposal_classifier_as_auxiliary"] = True
DRQN_with_prop_small["learning_rate_classifier"] = 0.000001

DRASCOS_no_rnn = DQN.copy()
DRASCOS["proposal_classifier"] = True
DRASCOS["scene_classifier"] = True

random_with_classifier = DRQN.copy()
random_with_classifier["proposal_classifier"] = True
random_with_classifier["scene_classifier"] = True
random_with_classifier["DQN"] = False

class replay_memory():
    def __init__(self, config):
        self.config = config
        self.memory_pos = []
        self.memory_neg = []
        self.pos_counter = 0
        self.neg_counter = 0
        self.max_memory_size = self.config["REPLAY_MEMORY"]
        self.current_train_length = 1

    def add(self, experience, positive=None):
        if self.config["episodic"]:
            if positive:
                if len(self.memory_pos) == self.max_memory_size:
                    self.memory_pos = self.memory_pos[1:]
                if len(experience) > self.config["MAX_RANGE_OF_RNN"]:
                    experience = experience[len(experience)-self.config["MAX_RANGE_OF_RNN"]:]
                self.memory_pos.append(experience)
                self.pos_counter += 1
            else:
                if len(self.memory_neg) == self.max_memory_size:
                    self.memory_neg = self.memory_neg[1:]
                if len(experience) > self.config["MAX_RANGE_OF_RNN"]:
                    experience = experience[len(experience)-self.config["MAX_RANGE_OF_RNN"]:]
                self.memory_neg.append(experience)
                self.neg_counter += 1
        else:
            if len(self.memory_pos) + len(experience) > self.max_memory_size:
                self.memory_pos = self.memory_pos[(len(self.memory_pos) + len(experience)) - self.max_memory_size:]
            self.memory_pos.extend(experience)
        #print(len(self.memory_pos))
        #print([len(l) for l in self.sampled_traces])
        #print(sum([len(l) for l in self.sampled_traces]))

    def check_if_enough_samples(self, batch_size, pos_ratio=None):
        if self.config["episodic"]:
            if pos_ratio is None:
                pos_ratio = self.pos_counter / (self.neg_counter + self.pos_counter)
            if len(self.memory_pos) < int(batch_size * 6 * pos_ratio):
                return False
            if len(self.memory_neg) < int(batch_size * 6 * (1 - pos_ratio)):
                return False
        else:
            if len(self.memory_pos) < int(batch_size):
                return False
        return True

    def sample(self, batch_size, pos_ratio=None):
        if self.config["episodic"]:
            if pos_ratio is None:
                pos_ratio = self.pos_counter/(self.neg_counter + self.pos_counter)
            sampled_traces_pos = [[] for _ in range(self.config["MAX_RANGE_OF_RNN"])]
            sampled_traces_neg = [[] for _ in range(self.config["MAX_RANGE_OF_RNN"])]
            sampled_episodes_pos = random.sample(self.memory_pos, int(batch_size * 6 * pos_ratio))
            sampled_episodes_neg = random.sample(self.memory_neg, int(batch_size * 6 * (1 - pos_ratio)))
            number_of_pos = batch_size * pos_ratio
            number_of_neg = batch_size - number_of_pos
            max_trace_length_pos = 1
            max_trace_length_neg = 1
            for idx in range(len(sampled_episodes_pos)):
                for trace_length in range(len(sampled_episodes_pos[idx])):
                    sampled_traces_pos[trace_length].append(idx)
                    if len(sampled_traces_pos[trace_length]) >= number_of_pos and max_trace_length_pos < trace_length + 1:
                        max_trace_length_pos = trace_length + 1
            for idx in range(len(sampled_episodes_neg)):
                for trace_length in range(len(sampled_episodes_neg[idx])):
                    sampled_traces_neg[trace_length].append(idx)
                    if len(sampled_traces_neg[trace_length]) >= number_of_neg and max_trace_length_neg < trace_length + 1:
                        max_trace_length_neg = trace_length + 1
            max_trace_length = min(max_trace_length_pos, max_trace_length_neg)
            trace_length = random.randint(1, max_trace_length)
            sample = []
            for idx in sampled_traces_pos[trace_length - 1]:
                sample.append(sampled_episodes_pos[idx][len(sampled_episodes_pos[idx])-trace_length:])
                if len(sample) == number_of_pos:
                    break
            for idx in sampled_traces_neg[trace_length - 1]:
                sample.append(sampled_episodes_neg[idx][len(sampled_episodes_neg[idx])-trace_length:])
                if len(sample) == batch_size:
                    break
            return sample, trace_length, number_of_pos, number_of_neg
        else:
            return random.sample(self.memory_pos, batch_size), 1, -1, -1

    def get_mean_episode_length(self):
        if self.config["episodic"]:
            lengths_pos = [len(ep) for ep in self.memory_pos]
            lengths_neg = [len(ep) for ep in self.memory_neg]
            return sum(lengths_pos) / len(lengths_pos), sum(lengths_neg) / len(lengths_neg)
        return 1, 1

    def get_pos_to_neg_ratio(self):
        return self.pos_counter/(self.neg_counter + self.pos_counter)

class Network:
    def __init__(self, agent_type, parameter_identifier, actions, postfix, use_bottlenecks, computing_on_cluster):
        self.actions = actions
        self.type = agent_type
        if self.type == 'DRQN':
            self.config = DRQN
        elif self.type == 'DRASCOS':
            self.config = DRASCOS
        elif self.type == 'DRASCOS_no_rnn':
            self.config = DRASCOS_no_rnn
        elif self.type == 'DQN':
            self.config = DQN
        elif self.type == 'random_with_classifier':
            self.config = random_with_classifier
        elif self.type == 'DRQN_with_auxiliary':
            self.config = DRQN_with_auxiliary
        elif self.type == 'DRQN_with_scene':
            self.config = DRQN_with_scene
        elif self.type == 'DRQN_with_prop':
            self.config = DRQN_with_prop
        elif self.type == 'DRQN_with_prop_small':
            self.config = DRQN_with_prop_small

        if self.config["proposal_classifier"] and not self.config["proposal_classifier_as_auxiliary"]:
            self.actions = actions - 1

        self.nof_output_modules = 0
        if self.config['DQN']:
            self.nof_output_modules += 2
        if self.config['proposal_classifier']:
            self.nof_output_modules += 1
        if self.config['scene_classifier']:
            self.nof_output_modules += 1
        self.postfix = postfix
        self.use_bottlenecks = use_bottlenecks
        self.computing_on_cluster = computing_on_cluster
        self.test_modus = False
        self.do_not_remember_this_episode = False
        self.current_state = None
        # init replay memory
        self.replay_memory = replay_memory(self.config)
        self.current_episode = []
        self.current_ep_terminated_positive = False
        self.scene_prediction = None
        self.proposal_prediction = None
        self.train_length = 0
        self.nof_pos = None
        self.nof_neg = None
        # init some parameters
        self.observe = True
        self.time_step = 0
        self.train_step = 0
        self.epsilon = self.config["INITIAL_EPSILON"]
        self.current_rnn_state_1 = None
        self.current_rnn_state_2 = None
        self.current_ep_scene_classifier_accuracy = 0.0
        self.hist_summaries = []
        self.train_summaries = []
        # init Q network
        self.inputs_outputs, self.QValue, self.proposal_class, self.scene_class, main_variables = self.createNetwork('main')

        # init Target Q Network
        self.inputs_outputsT, self.QValueT, _, _, target_variables = self.createNetwork('target')

        self.copyTargetQNetworkOperation = []
        for idx in range(len(main_variables)):
            self.copyTargetQNetworkOperation.append(
                target_variables[idx].assign(main_variables[idx]))

        if self.config['unified_aggregate']:
            self.main_lstm_variables = [v for v in tf.global_variables() if v.name.startswith('main_rnn')]
            self.target_lstm_variables = [v for v in tf.global_variables() if v.name.startswith('target_rnn')]
        else:
            self.main_lstm_variables = [v for v in tf.global_variables() if v.name.startswith('main_rnn_rl')]
            self.target_lstm_variables = [v for v in tf.global_variables() if v.name.startswith('target_rnn_rl')]
            self.main_lstm_variables.extend([v for v in tf.global_variables() if v.name.startswith('main_rnn_sl')])
            self.target_lstm_variables.extend([v for v in tf.global_variables() if v.name.startswith('target_rnn_sl')])
        for idx in range(len(self.main_lstm_variables)):
            self.copyTargetQNetworkOperation.append(self.target_lstm_variables[idx].assign(self.main_lstm_variables[idx]))

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()

        summary_path = 'trainsummary' + self.postfix
        self.train_summary_writer = tf.summary.FileWriter(summary_path + '/train')
        self.test_summary_writer = tf.summary.FileWriter(summary_path + '/test')

        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("./savedweights" + self.postfix)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            with open("./savedweights" + self.postfix + "/checkpoint", 'r') as file:
                for line in file:
                    line = line.strip()
                    line = line.split('-')
                step = line[2][:-1]
            self.train_step = int(step)
            print("Successfully loaded:", checkpoint.model_checkpoint_path, "at train step:", self.train_step)
        else:
            print("Could not find old network weights")
            self.train_summary_writer.add_graph(self.session.graph)

    def createNetwork(self, my_scope):
        variables = []
        unified_aggregate = self.config['unified_aggregate']
        nof_output_modules = self.nof_output_modules

        keep_prob = tf.placeholder(tf.float32)
        keep_prob_per_input = tf.placeholder(tf.float32)
        image_noise = tf.placeholder(tf.float32)
        object_detector_noise = tf.placeholder(tf.float32)
        trainLength = tf.placeholder(dtype=tf.int32)
        batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        if not self.use_bottlenecks:
            viewInput = tf.placeholder("float", [None, 338, 600, 3])
            #  normalize image
            viewInput_normalized = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), viewInput)
        else:
            viewInput = tf.placeholder("float", [None, 2048])
            viewInput_normalized = viewInput
        if self.config["episodic"]:
            objectInput = tf.placeholder("float", [None, 33, 5])
            objectInput_flat = tf.reshape(objectInput, [-1, 165])
        else:
            objectInput = tf.placeholder("float", [None, 33, 5, 3])
            objectInput_flat = tf.reshape(objectInput, [-1, 495])
        allowedActionsInput = tf.placeholder("float", [None, self.actions])
        objectProposalInput = tf.placeholder("float", [None, 2048])

        #  add noise to image and object detector input

        viewInput_noisy = self.gaussian_noise_layer(viewInput_normalized, image_noise)
        objectProposalInput_noisy = self.gaussian_noise_layer(objectProposalInput, image_noise)
        objectInput_flat_noisy = self.gaussian_noise_layer(objectInput_flat, object_detector_noise)

        if not self.use_bottlenecks:
            with tf.name_scope(my_scope + "/conv_layer_1"):
                W_conv0 = self.weight_variable([10, 12, 3, 16])
                b_conv0 = self.bias_variable([16])
                variables.append(W_conv0)
                variables.append(b_conv0)
                conv_layer_1_act = tf.nn.relu(self.conv2d(viewInput_noisy, W_conv0, 4) + b_conv0)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_conv0)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_conv0)]
                    # tf.summary.histogram("activations", conv_layer_1_act)
            with tf.name_scope(my_scope + "/conv_layer_2"):
                W_conv1 = self.weight_variable([7, 8, 16, 32])
                b_conv1 = self.bias_variable([32])
                variables.append(W_conv1)
                variables.append(b_conv1)
                conv_layer_2_act = tf.nn.relu(self.conv2d(conv_layer_1_act, W_conv1, 4) + b_conv1)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_conv1)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_conv1)]
                    # tf.summary.histogram("activations", conv_layer_2_act)
            with tf.name_scope(my_scope + "/conv_layer_3"):
                W_conv2 = self.weight_variable([6, 6, 32, 64])
                b_conv2 = self.bias_variable([64])
                variables.append(W_conv2)
                variables.append(b_conv2)
                conv_layer_3_act = tf.nn.relu(self.conv2d(conv_layer_2_act, W_conv2, 2) + b_conv2)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_conv2)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_conv2)]
                    # tf.summary.histogram("activations", conv_layer_3_act)
            with tf.name_scope(my_scope + "/conv_layer_4"):
                W_conv3 = self.weight_variable([3, 3, 64, 64])
                b_conv3 = self.bias_variable([64])
                variables.append(W_conv3)
                variables.append(b_conv3)
                conv_layer_4_act = tf.nn.relu(self.conv2d(conv_layer_3_act, W_conv3, 1) + b_conv3)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_conv3)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_conv3)]
                    # tf.summary.histogram("activations", conv_layer_4_act)
            conv_layer_4_act_flat = tf.reshape(conv_layer_4_act, [-1, 5376])
            with tf.name_scope(my_scope + "/fc_layer_view_image"):
                W_v_fc1 = self.weight_variable([5376, 512])
                b_v_fc1 = self.bias_variable([512])
                variables.append(W_v_fc1)
                variables.append(b_v_fc1)
                fc_layer_view_image_act = tf.nn.relu(tf.matmul(conv_layer_4_act_flat, W_v_fc1) + b_v_fc1)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_v_fc1)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_v_fc1)]
                    # tf.summary.histogram("activations", fc_layer_view_image_act)
        else:
            with tf.name_scope(my_scope + "/fc_layer_view_image"):
                W_v_fc1 = self.weight_variable([2048, 512])
                b_v_fc1 = self.bias_variable([512])
                variables.append(W_v_fc1)
                variables.append(b_v_fc1)
                fc_layer_view_image_act = tf.nn.relu(tf.matmul(viewInput_noisy, W_v_fc1) + b_v_fc1)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_v_fc1)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_v_fc1)]
                    # tf.summary.histogram("activations", fc_layer_view_image_act)

        with tf.name_scope(my_scope + "/fc_layer_od_1"):
            if self.config["episodic"]:
                W_o_fc1 = self.weight_variable([165, 512])
            else:
                W_o_fc1 = self.weight_variable([495, 512])
            b_o_fc1 = self.bias_variable([512])
            variables.append(W_o_fc1)
            variables.append(b_o_fc1)
            fc_layer_od_1_act = tf.nn.relu(tf.matmul(objectInput_flat_noisy, W_o_fc1) + b_o_fc1)
            if self.config["log_histogram"]:
                self.hist_summaries += [tf.summary.histogram("hidden_weights", W_o_fc1)]
                self.hist_summaries += [tf.summary.histogram("bias", b_o_fc1)]
                # tf.summary.histogram("activations", fc_layer_od_1_act)
        with tf.name_scope(my_scope + "/fc_layer_od_2"):
            W_o_fc2 = self.weight_variable([512, 512])
            b_o_fc2 = self.bias_variable([512])
            variables.append(W_o_fc2)
            variables.append(b_o_fc2)
            fc_layer_od_2_act = tf.nn.relu(tf.matmul(fc_layer_od_1_act, W_o_fc2) + b_o_fc2)
            if self.config["log_histogram"]:
                self.hist_summaries += [tf.summary.histogram("hidden_weights", W_o_fc2)]
                self.hist_summaries += [tf.summary.histogram("bias", b_o_fc2)]
                # tf.summary.histogram("activations", fc_layer_od_2_act)

        with tf.name_scope(my_scope + "/fc_layer_object_proposal"):
            W_p_fc1 = self.weight_variable([2048, 512])
            b_p_fc1 = self.bias_variable([512])
            variables.append(W_p_fc1)
            variables.append(b_p_fc1)
            fc_layer_object_proposal_act = tf.nn.relu(tf.matmul(objectProposalInput_noisy, W_p_fc1) + b_p_fc1)
            if self.config["log_histogram"]:
                self.hist_summaries += [tf.summary.histogram("hidden_weights", W_p_fc1)]
                self.hist_summaries += [tf.summary.histogram("bias", b_p_fc1)]
                # tf.summary.histogram("activations", fc_layer_object_proposal_act)


        merged = tf.reshape(tf.concat([fc_layer_view_image_act, fc_layer_od_2_act, fc_layer_object_proposal_act], axis=1),
                            shape=[trainLength*batch_size, 3, 512])
        # drop entire input?
        merged = tf.nn.dropout(merged, keep_prob=keep_prob_per_input, noise_shape=[trainLength*batch_size, 1, 512])
        merged = tf.reshape(merged, shape=[trainLength*batch_size, 1536])
        # drop element-wise
        merged = tf.nn.dropout(merged, keep_prob=keep_prob)

        #self.log_scalar('sparsity', tf.nn.zero_fraction(merged))

        with tf.name_scope(my_scope + "/fc_layer_before_rnn_1"):
            W_bf1 = self.weight_variable([1536, 1536])
            b_bf1 = self.bias_variable([1536])
            variables.append(W_bf1)
            variables.append(b_bf1)
            before_rnn = tf.nn.relu(tf.matmul(merged, W_bf1) + b_bf1)
            if self.config["log_histogram"]:
                self.hist_summaries += [tf.summary.histogram("hidden_weights", W_bf1)]
                self.hist_summaries += [tf.summary.histogram("bias", b_bf1)]

        rnn_state_in_1 = None
        rnn_state_out_1 = None
        rnn_state_in_2 = None
        rnn_state_out_2 = None
        if self.config["RNN_LAYERS"] > 0:
            if unified_aggregate:
                with tf.name_scope(my_scope + "/rnn_layers"):
                    fc_0 = tf.reshape(before_rnn, [batch_size, trainLength, 1536])
                    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True) for size in
                                  [1536] * self.config["RNN_LAYERS"]]
                    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

                    rnn_state_in_1 = rnn_cell.zero_state(batch_size, tf.float32)
                    aggregate, rnn_state_out_1 = tf.nn.dynamic_rnn(
                        inputs=fc_0, cell=rnn_cell, dtype="float", initial_state=rnn_state_in_1, scope=my_scope + '_rnn')
                    aggregate = tf.reshape(aggregate, shape=[-1, 1536])
            else:
                with tf.name_scope(my_scope + "/rnn_layers_RL"):
                    fc_0_rl = tf.reshape(before_rnn, [batch_size, trainLength, 1536])
                    rnn_layers_rl = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True) for size in
                                  [1536] * self.config["RNN_LAYERS"]]
                    rnn_cell_rl = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_rl)

                    rnn_state_in_1 = rnn_cell_rl.zero_state(batch_size, tf.float32)
                    aggregate_rl, rnn_state_out_1 = tf.nn.dynamic_rnn(
                        inputs=fc_0_rl, cell=rnn_cell_rl, dtype="float", initial_state=rnn_state_in_1, scope=my_scope + '_rnn_rl')
                    aggregate_rl = tf.reshape(aggregate_rl, shape=[-1, 1536])
                with tf.name_scope(my_scope + "/rnn_layers_SL"):
                    fc_0_sl = tf.reshape(before_rnn, [batch_size, trainLength, 1536])
                    rnn_layers_sl = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True) for size in
                                  [1536] * self.config["RNN_LAYERS"]]
                    rnn_cell_sl = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_sl)

                    rnn_state_in_2 = rnn_cell_sl.zero_state(batch_size, tf.float32)
                    aggregate_sl, rnn_state_out_2 = tf.nn.dynamic_rnn(
                        inputs=fc_0_sl, cell=rnn_cell_sl, dtype="float", initial_state=rnn_state_in_2, scope=my_scope + '_rnn_sl')
                    aggregate_sl = tf.reshape(aggregate_sl, shape=[-1, 1536])

        else:
            with tf.name_scope(my_scope + "/replacement_rnn_layer"):
                W_agg = self.weight_variable([1536, 1536])
                b_agg = self.bias_variable([1536])
                variables.append(W_agg)
                variables.append(b_agg)
                aggregate = tf.nn.relu(tf.matmul(before_rnn, W_agg) + b_agg)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_agg)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_agg)]

        splits = []
        if unified_aggregate:
            with tf.name_scope(my_scope + "/fc_layer_after_rnn_1"):
                W_fc1 = self.weight_variable([1536, 1024 * nof_output_modules])
                b_fc1 = self.bias_variable([1024 * nof_output_modules])
                variables.append(W_fc1)
                variables.append(b_fc1)
                fc_layer_after_rnn_1 = tf.nn.relu(tf.matmul(aggregate, W_fc1) + b_fc1)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_fc1)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_fc1)]
            splits = tf.split(fc_layer_after_rnn_1, nof_output_modules, 1)
        else:
            with tf.name_scope(my_scope + "/fc_layer_after_rnn_rl_1"):
                W_fc1_rl = self.weight_variable([1536, 1024 * 2])
                b_fc1_rl = self.bias_variable([1024 * 2])
                variables.append(W_fc1_rl)
                variables.append(b_fc1_rl)
                fc_layer_after_rnn_rl_1 = tf.nn.relu(tf.matmul(aggregate_rl, W_fc1_rl) + b_fc1_rl)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_fc1_rl)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_fc1_rl)]
            splits.extend(tf.split(fc_layer_after_rnn_rl_1, 2, 1))
            with tf.name_scope(my_scope + "/fc_layer_after_rnn_sl_1"):
                W_fc1_sl = self.weight_variable([1536, 1024 * 2])
                b_fc1_sl = self.bias_variable([1024 * 2])
                variables.append(W_fc1_sl)
                variables.append(b_fc1_sl)
                fc_layer_after_rnn_sl_1 = tf.nn.relu(tf.matmul(aggregate_sl, W_fc1_sl) + b_fc1_sl)
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_fc1_sl)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_fc1_sl)]
            splits.extend(tf.split(fc_layer_after_rnn_sl_1, 2, 1))

        current_split = 0
        if self.config['DQN']:
            Adv_split = splits[current_split]
            Val_split = splits[current_split + 1]
            current_split += 2
        if self.config['proposal_classifier']:
            proposal_class_split = splits[current_split]
            current_split += 1
        if self.config['scene_classifier']:
            scene_class_split = splits[current_split]
            current_split += 1

        QValue = None
        if self.config['DQN']:
            Adv_split = tf.reshape(tf.concat([Adv_split, allowedActionsInput], axis=1), [-1, 1024 + self.actions])
            with tf.name_scope(my_scope + "/rl_layer_advantage"):
                W_fc_Adv = self.weight_variable([1024 + self.actions, self.actions])
                b_fc_Adv = self.bias_variable([self.actions])
                variables.append(W_fc_Adv)
                variables.append(b_fc_Adv)
                Advantage = tf.matmul(Adv_split, W_fc_Adv) + b_fc_Adv
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_fc_Adv)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_fc_Adv)]
            with tf.name_scope(my_scope + "/rl_layer_advantage"):
                W_fc_Val = self.weight_variable([1024, 1])
                b_fc_Val = self.bias_variable([1])
                variables.append(W_fc_Val)
                variables.append(b_fc_Val)
                Value = tf.matmul(Val_split, W_fc_Val) + b_fc_Val
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_fc_Val)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_fc_Val)]

            # Advantage = tf.Print(Advantage, [Advantage], message="This is the Advantage: ", summarize=8)
            mean_Advantage = tf.reduce_mean(Advantage, axis=1, keep_dims=True)
            # mean_Advantage = tf.Print(mean_Advantage, [mean_Advantage], message="This is the mean_Advantage: ", summarize=8)
            # Value = tf.Print(Value, [Value], message="This is the Value: ", summarize=8)
            QValue = Value + tf.subtract(Advantage, mean_Advantage)
            # QValue = tf.Print(QValue, [QValue], message="This is the QValue: ", summarize=8)
            # QValue = tf.matmul(h_fc4, W_fcQ) + b_fcQ

        proposal_class = None
        if self.config['proposal_classifier']:
            with tf.name_scope(my_scope + "/classification_layer_proposal"):
                W_fcC = self.weight_variable([1024, 2])
                b_fcC = self.bias_variable([2])
                variables.append(W_fcC)
                variables.append(b_fcC)
                proposal_class = tf.matmul(proposal_class_split, W_fcC) + b_fcC
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_fcC)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_fcC)]

        scene_class = None
        if self.config['scene_classifier']:
            with tf.name_scope(my_scope + "/classification_layer_scene"):
                W_fcS = self.weight_variable([1024, 28])
                b_fcS = self.bias_variable([28])
                variables.append(W_fcS)
                variables.append(b_fcS)
                scene_class = tf.matmul(scene_class_split, W_fcS) + b_fcS
                if self.config["log_histogram"]:
                    self.hist_summaries += [tf.summary.histogram("hidden_weights", W_fcS)]
                    self.hist_summaries += [tf.summary.histogram("bias", b_fcS)]

        if self.config["log_histogram"]:
            self.merged_hist_summary = tf.summary.merge(self.hist_summaries)

        return [viewInput, objectInput, rnn_state_in_1, rnn_state_out_1, trainLength, batch_size, allowedActionsInput,
                keep_prob, keep_prob_per_input, image_noise, object_detector_noise, objectProposalInput, rnn_state_in_2, rnn_state_out_2], \
                QValue, proposal_class, scene_class, variables

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        if self.config["DQN"]:
            # this is an array like [0,0,0,1,0,0]
            self.actionInput = tf.placeholder("float", [None, self.actions])
            self.yInput = tf.placeholder("float", [None])
            # compute predicted value for action times the action took
            # the following reduction gives us the predicted reward for the chosen action for each training sample
            Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
            self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
            #tf.summary.scalar('training_mean_Q-value', tf.reduce_mean(Q_Action))
            self.train_summaries += [tf.summary.scalar('mean_squared_Q-value_training_loss', self.cost)]
            # we want to minimize difference between actual and predicted reward
            # actual reward is computed with r + discount factor x predicted reward for next action
            #self.train_dqn = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)
            self.train_dqn = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"]).minimize(self.cost)
        if self.config["proposal_classifier"]:
            self.proposal_labels = tf.placeholder("float", [None, 2])
            #weights = [1] * self.config["BATCH_SIZE"]
            #weights = tf.reshape(tf.constant([weights], "float"), [-1])
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.proposal_labels, logits=self.proposal_class)
                                                            #weights=weights)
            self.train_summaries += [tf.summary.scalar('proposal_classifier_cross_entropy', cross_entropy)]
            self.train_proposal_classifier = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate_classifier"]).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self.proposal_class, 1), tf.argmax(self.proposal_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #tf.summary.scalar('proposal_classifier_accuracy', accuracy)

            self.mask_values_pos = tf.placeholder(tf.int64, [None])
            self.mask_values_neg = tf.placeholder(tf.int64, [None])

            #should_have_mask = tf.equal(self.mask_values_pos, tf.argmax(self.proposal_labels, 1))
            #positive_split = tf.boolean_mask(correct_prediction, should_have_mask)
            #should_not_have_mask = tf.equal(self.mask_values_neg, tf.argmax(self.proposal_labels, 1))
            #negative_split = tf.boolean_mask(correct_prediction, should_not_have_mask)
            #pos_accuracy = tf.reduce_mean(tf.cast(positive_split, tf.float32))
            #tf.summary.scalar('proposal_classifier_pos_accuracy', pos_accuracy)
            #neg_accuracy = tf.reduce_mean(tf.cast(negative_split, tf.float32))
            #tf.summary.scalar('proposal_classifier_neg_accuracy', neg_accuracy)
        if self.config["scene_classifier"]:
            self.scene_labels = tf.placeholder("float", [None, 28])
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.scene_labels, logits=self.scene_class)
            self.train_summaries += [tf.summary.scalar('scene_classifier_cross_entropy', cross_entropy)]
            self.train_scene_classifier = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate_classifier"]).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self.scene_class, 1), tf.argmax(self.scene_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_summaries += [tf.summary.scalar('scene_classifier_training_accuracy', accuracy)]

        self.merged_train_summary = tf.summary.merge(self.train_summaries)
        #self.merged_summary = tf.summary.merge_all()

    def trainNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch, self.train_length, self.nof_pos, self.nof_neg = self.replay_memory.sample(self.config['BATCH_SIZE'], self.config['pos_ratio'])
        #self.log_scalar('ep_train_length', self.train_length)
        #print("current train length: ", self.train_length)
        #self.log_scalar('sampled_pos_ratio', self.nof_pos/(self.nof_pos+self.nof_neg))
        mean_episode_lengths_pos, mean_episode_lengths_neg = self.replay_memory.get_mean_episode_length()
        self.log_scalar('mean_episode_lengths_in_memomory_pos', mean_episode_lengths_pos)
        #self.log_scalar('mean_episode_lengths_in_memomory_neg', mean_episode_lengths_neg)
        #self.log_scalar('pos_to_neg_ratio', self.replay_memory.get_pos_to_neg_ratio())

        if self.config['episodic']:
            view_batch = [exp[0][0] for ep in minibatch for exp in ep]
            objects_batch = [exp[0][1] for ep in minibatch for exp in ep]
            if self.config["proposal_classifier"] and not self.config["proposal_classifier_as_auxiliary"]:
                allowedActions_batch = [exp[0][2][1:] for ep in minibatch for exp in ep]
            else:
                allowedActions_batch = [exp[0][2] for ep in minibatch for exp in ep]
            object_proposals_batch = [exp[0][3] for ep in minibatch for exp in ep]
            if self.config["proposal_classifier"] and not self.config["proposal_classifier_as_auxiliary"]:
                action_batch = [exp[1][1:] for ep in minibatch for exp in ep]
            else:
                action_batch = [exp[1] for ep in minibatch for exp in ep]
            reward_batch = [exp[2] for ep in minibatch for exp in ep]
            nextView_batch = [exp[3][0] for ep in minibatch for exp in ep]
            nextObjects_batch = [exp[3][1] for ep in minibatch for exp in ep]
            if self.config["proposal_classifier"] and not self.config["proposal_classifier_as_auxiliary"]:
                nextAllowedActions_batch = [exp[3][2][1:] for ep in minibatch for exp in ep]
            else:
                nextAllowedActions_batch = [exp[3][2] for ep in minibatch for exp in ep]
            nextObject_proposals_batch = [exp[3][3] for ep in minibatch for exp in ep]
            terminal = [exp[4] for ep in minibatch for exp in ep]
            proposal_labels_batch = [exp[5] for ep in minibatch for exp in ep]
            scene_labels_batch = [exp[6] for ep in minibatch for exp in ep]
        else:
            view_batch = [exp[0][0] for exp in minibatch]
            objects_batch = [exp[0][1] for exp in minibatch]
            allowedActions_batch = [exp[0][2] for exp in minibatch]
            object_proposals_batch = [exp[0][3] for exp in minibatch]
            action_batch = [exp[1] for exp in minibatch]
            reward_batch = [exp[2] for exp in minibatch]
            nextView_batch = [exp[3][0] for exp in minibatch]
            nextObjects_batch = [exp[3][1] for exp in minibatch]
            nextAllowedActions_batch = [exp[3][2] for exp in minibatch]
            nextObject_proposals_batch = [exp[3][3] for exp in minibatch]
            terminal = [exp[4] for exp in minibatch]
            proposal_labels_batch = [exp[5] for exp in minibatch]
            scene_labels_batch = [exp[6] for exp in minibatch]

        if self.config["RNN_LAYERS"] > 0:
            state_train_1 = tuple([np.zeros([self.config["BATCH_SIZE"], 1536]) for _ in range(self.config["RNN_LAYERS"] * 2)])
            if not self.config["unified_aggregate"]:
                state_train_2 = tuple([np.zeros([self.config["BATCH_SIZE"], 1536]) for _ in range(self.config["RNN_LAYERS"] * 2)])

        if self.config['DQN']:
            input_dict = {
                self.inputs_outputsT[0]: nextView_batch,
                self.inputs_outputsT[1]: nextObjects_batch,
                self.inputs_outputsT[4]: self.train_length,
                self.inputs_outputsT[5]: self.config["BATCH_SIZE"],
                self.inputs_outputsT[6]: nextAllowedActions_batch,
                self.inputs_outputsT[7]: 1.0,
                self.inputs_outputsT[8]: 1.0,
                self.inputs_outputsT[9]: 0.0,
                self.inputs_outputsT[10]: 0.0,
                self.inputs_outputsT[11]: nextObject_proposals_batch,
            }
            if self.config["RNN_LAYERS"] > 0:
                input_dict[self.inputs_outputsT[2]] = state_train_1
                if not self.config["unified_aggregate"]:
                    input_dict[self.inputs_outputsT[12]] = state_train_2

            y_batch = []
            QValue_batch = self.QValueT.eval(feed_dict=input_dict)
            for i in range(0, self.config["BATCH_SIZE"] * self.train_length):
                if terminal[i]:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + self.config["GAMMA"] * np.max(QValue_batch[i]))


        if self.train_step % self.config["histogram_log_time"] == 0 and self.config["log_histogram"]:
            merged_summary = self.merged_hist_summary
        else:
            merged_summary = self.merged_train_summary

        train_ops = [merged_summary]
        if self.config['DQN']:
            train_ops.append(self.train_dqn)
        if self.config['proposal_classifier']:
            train_ops.append(self.train_proposal_classifier)
        if self.config['scene_classifier']:
            train_ops.append(self.train_scene_classifier)

        keep_probs = [1.0] * 2

        # train with all three inputs in 2/3 of all training steps
        # train with two of the three inputs in 2/9 of all training steps
        # train only one input in 1/9 of all training steps
        #if random.random() < 1/3:
        #    # if we drop an entire input the other inputs are fully provided
        #    idx_list = [1, 2, 3]
        #    if random.random() < 2/3:
        #        idx_sample = [idx_list[i] for i in sorted(random.sample(range(len(idx_list)), 2))]
        #    else:
        #        idx_sample = [idx_list[i] for i in sorted(random.sample(range(len(idx_list)), 1))]
        #    for idx in idx_sample:
        #        keep_probs[idx] = np.nextafter(0, 1)
        #else:
        #    keep_probs[0] = self.config["element_wise_keep_prob"]


        # with 50% chance train with all inputs but dropout rate per node
        # in other cases train with chance of input dropout
        if random.random() < 1/2:
            keep_probs[0] = self.config["element_wise_keep_prob_for_input_drop"]
            keep_probs[1] = self.config["input_keep_prob"]
        else:
            keep_probs[0] = self.config["element_wise_keep_prob"]

        input_dict = {
            self.inputs_outputs[0]: view_batch,
            self.inputs_outputs[1]: objects_batch,
            self.inputs_outputs[4]: self.train_length,
            self.inputs_outputs[5]: self.config["BATCH_SIZE"],
            self.inputs_outputs[6]: allowedActions_batch,
            self.inputs_outputs[7]: keep_probs[0],
            self.inputs_outputs[8]: keep_probs[1],
            self.inputs_outputs[9]: self.config["image_noise"],
            self.inputs_outputs[10]: self.config["object_detector_noise"],
            self.inputs_outputs[11]: object_proposals_batch,
        }
        if self.config['DQN']:
            input_dict[self.actionInput] = action_batch
            input_dict[self.yInput] = y_batch
        if self.config["RNN_LAYERS"] > 0:
            input_dict[self.inputs_outputs[2]] = state_train_1
            if not self.config["unified_aggregate"]:
                input_dict[self.inputs_outputs[12]] = state_train_2
        if self.config["proposal_classifier"]:
            input_dict[self.proposal_labels] = proposal_labels_batch
            input_dict[self.mask_values_pos] = np.zeros(self.config["BATCH_SIZE"] * self.train_length)
            input_dict[self.mask_values_neg] = np.ones(self.config["BATCH_SIZE"] * self.train_length)
        if self.config["scene_classifier"]:
            input_dict[self.scene_labels] = scene_labels_batch

        outputs = self.session.run(train_ops, feed_dict=input_dict)

        self.train_summary_writer.add_summary(outputs[0], self.train_step)
        self.train_summary_writer.flush()

        self.train_step += 1
        if self.train_step % self.config["SAVE_TIME"] == 0:
            print("Saving")
            self.saver.save(self.session, './savedweights' + self.postfix + '/network' + '-dqn', global_step=self.train_step)

        if self.train_step % self.config["UPDATE_TIME"] == 0:
            self.copyTargetQNetwork()

    def getAction(self):
        # change episilon
        if self.epsilon > self.config["FINAL_EPSILON"] and not self.observe:
            self.epsilon -= (self.config["INITIAL_EPSILON"] - self.config["FINAL_EPSILON"]) / self.config["EXPLORE"]

        allowed_actions = self.current_state[2]
        if self.config["proposal_classifier"] and not self.config["proposal_classifier_as_auxiliary"]:
            allowed_actions = allowed_actions[1:]

        input_dict = {
            self.inputs_outputs[0]: [self.current_state[0]],
            self.inputs_outputs[1]: [self.current_state[1]],
            self.inputs_outputs[4]: 1,
            self.inputs_outputs[5]: 1,
            self.inputs_outputs[6]: [allowed_actions],  # allowed actions
            self.inputs_outputs[7]: 1.0,
            self.inputs_outputs[8]: 1.0,
            self.inputs_outputs[9]: 0.0,
            self.inputs_outputs[10]: 0.0,
            self.inputs_outputs[11]: [self.current_state[3]],

        }
        action_ops = []
        if self.config['DQN']:
            action_ops.append(self.QValue)
        if self.config['proposal_classifier'] and not self.config["proposal_classifier_as_auxiliary"]:
            action_ops.append(self.proposal_class)
        if self.config['scene_classifier']:
            action_ops.append(self.scene_class)
        if self.config["RNN_LAYERS"] > 0:
            action_ops.append(self.inputs_outputs[3])
            input_dict[self.inputs_outputs[2]] = self.current_rnn_state_1
            if not self.config["unified_aggregate"]:
                action_ops.append(self.inputs_outputs[13])
                input_dict[self.inputs_outputs[12]] = self.current_rnn_state_2

        outputs = self.session.run(action_ops, feed_dict=input_dict)

        action = np.zeros(7)
        current_output = 0
        return_values = []
        if self.config['DQN']:
            QValue = outputs[current_output]
            QValue = QValue[0]
            current_output += 1
        if self.config['proposal_classifier'] and not self.config["proposal_classifier_as_auxiliary"]:
            self.proposal_prediction = outputs[current_output]
            self.proposal_prediction = self.proposal_prediction[0]
            Class = self.softmax(self.proposal_prediction)
            if Class[0] > Class[1]:
                action[0] = 1
            return_values.append(Class[0])
            current_output += 1
        if self.config['scene_classifier']:
            self.scene_prediction = outputs[current_output]
            self.scene_prediction = self.scene_prediction[0]
            current_output += 1
        if self.config["RNN_LAYERS"] > 0:
            self.current_rnn_state_1 = outputs[current_output]
            current_output += 1
            if not self.config["unified_aggregate"]:
                self.current_rnn_state_2 = outputs[current_output]
            current_output += 1

        if action[0] == 0:
            if self.config['DQN']:
                if random.random() <= self.epsilon:
                    action_index = random.randrange(self.actions)
                    action[action_index+7-self.actions] = 1
                else:
                    action_index = np.argmax(QValue)
                    action[action_index+7-self.actions] = 1
                return_values.extend(self.softmax(QValue).tolist())
            else:
                while True:
                    action_index = random.randrange(self.actions)
                    if self.current_state[2][1:][action_index] == 1:
                        break
                action[action_index + 7 - self.actions] = 1
                return_values.extend([0] * self.actions)

        return action, return_values

    def setPerception(self, next_observation, action, reward, terminal, proposal_label, scene_label):
        new_state = self.get_new_state(next_observation)
        self.current_episode.append([self.current_state, action, reward, new_state, terminal, proposal_label, scene_label])
        self.current_state = new_state
        if terminal:
            if action[0] == 1 and proposal_label[0] == 1:
                self.current_ep_terminated_positive = True
            else:
                self.current_ep_terminated_positive = False
            self.log_scalar('success', float(self.current_ep_terminated_positive))
            if self.current_ep_terminated_positive:
                self.log_scalar('scene_classifier_pos_episode_accuracy',
                                self.current_ep_scene_classifier_accuracy / len(self.current_episode))
            else:
                self.log_scalar('scene_classifier_neg_episode_accuracy',
                                self.current_ep_scene_classifier_accuracy / len(self.current_episode))
            self.current_ep_scene_classifier_accuracy = 0.0
            if self.current_ep_terminated_positive:
                self.log_scalar('episode_length_pos', len(self.current_episode))
            else:
                self.log_scalar('episode_length_neg', len(self.current_episode))

        if not self.test_modus:
            if self.observe:
                self.observe = not self.replay_memory.check_if_enough_samples(self.config['BATCH_SIZE'], self.config['pos_ratio'])
            if not self.observe and self.time_step % self.config["TRAIN_EVERY"] == 0:
                # Train the network
                self.trainNetwork()
            if self.observe:
                state = "observe"
            elif not self.observe and self.time_step <= self.config["EXPLORE"]:
                state = "explore"
            else:
                state = "train"
            if self.time_step % 5000 == 0:
                print("TIMESTEP", self.time_step, "/ STATE", state, "/ EPSILON", self.epsilon)
            self.time_step += 1

        if self.config["proposal_classifier"]:
            if np.argmax(self.proposal_prediction) == np.argmax(proposal_label):
                #self.log_scalar('proposal_classifier_accuracy', 1)
                if np.argmax(proposal_label) == 0:
                    self.log_scalar('proposal_classifier_pos_accuracy', 1)
                else:
                    self.log_scalar('proposal_classifier_neg_accuracy', 1)
            else:
                # self.log_scalar('proposal_classifier_accuracy', 0)
                if np.argmax(proposal_label) == 0:
                    self.log_scalar('proposal_classifier_pos_accuracy', 0)
                else:
                    self.log_scalar('proposal_classifier_neg_accuracy', 0)

        if self.config["scene_classifier"]:
            if np.argmax(self.scene_prediction) == np.argmax(scene_label):
                self.current_ep_scene_classifier_accuracy += 1

    def get_new_state(self, next_observation):
        if self.config["episodic"]:
            return next_observation
        else:
            return self.append_observation(self.current_state, next_observation)

    def append_observation(self, old_obs, new_obs):
        previousState = old_obs[1]
        nextView = np.expand_dims(new_obs[1], axis=2)
        return [new_obs[0], np.append(previousState[:, :, 1:], nextView, axis=2), new_obs[2], new_obs[3]]

    def setInitState(self, observation):
        if self.config["episodic"]:
            self.current_rnn_state_1 = tuple([np.zeros([1, 1536]) for _ in range(self.config["RNN_LAYERS"]*2)])
            if not self.config["unified_aggregate"]:
                self.current_rnn_state_2 = tuple([np.zeros([1, 1536]) for _ in range(self.config["RNN_LAYERS"] * 2)])
            self.current_state = observation
        else:
            self.current_state = [observation[0], np.stack((observation[1], observation[1], observation[1]), axis=2), observation[2], observation[3]]
        if len(self.current_episode) > 0 and not self.do_not_remember_this_episode:
            self.replay_memory.add(self.current_episode, self.current_ep_terminated_positive)
        if self.test_modus:
            self.do_not_remember_this_episode = True
        else:
            self.do_not_remember_this_episode = False
        self.current_episode = []

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def set_test_modus(self, test_modus):
        self.test_modus = test_modus

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def log_scalar(self, tag, value):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        if self.test_modus:
            self.test_summary_writer.add_summary(summary, self.train_step)
            self.test_summary_writer.flush()
        else:
            self.train_summary_writer.add_summary(summary, self.train_step)
            self.train_summary_writer.flush()

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return tf.add(input_layer, tf.multiply(input_layer, noise))

    def createDRQNNetwork(self, my_scope):
        W_conv0 = self.weight_variable([10, 12, 3, 16])
        b_conv0 = self.bias_variable([16])

        W_conv1 = self.weight_variable([7, 8, 16, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([6, 6, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_v_fc1 = self.weight_variable([5376, 1024])
        b_v_fc1 = self.bias_variable([1024])

        #######

        W_o_fc1 = self.weight_variable([165, 512])
        b_o_fc1 = self.bias_variable([512])

        W_o_fc2 = self.weight_variable([512, 512])
        b_o_fc2 = self.bias_variable([512])

        #######

        W_fc1 = self.weight_variable([1536, 1024])
        b_fc1 = self.bias_variable([1024])

        # W_fc2 = self.weight_variable([1980, 1980])
        # b_fc2 = self.bias_variable([1980])

        # W_fc3 = self.weight_variable([1980, 1980])
        # b_fc3 = self.bias_variable([1980])

        # W_fc4 = self.weight_variable([1980, 990])
        # b_fc4 = self.bias_variable([990])

        W_fc_Val = self.weight_variable([512, 1])
        b_fc_Val = self.bias_variable([1])

        W_fc_Adv = self.weight_variable([512 + self.actions, self.actions])
        b_fc_Adv = self.bias_variable([self.actions])

        keep_prob = tf.placeholder(tf.float32)
        trainLength = tf.placeholder(dtype=tf.int32)
        batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        viewInput = tf.placeholder("float", [None, 338, 600, 3])
        objectInput = tf.placeholder("float", [None, 33, 5])
        objectInput_flat = tf.reshape(objectInput, [-1, 165])
        allowedActionsInput = tf.placeholder("float", [None, self.actions])

        h_conv0 = tf.nn.relu(self.conv2d(viewInput, W_conv0, 4) + b_conv0)
        h_conv1 = tf.nn.relu(self.conv2d(h_conv0, W_conv1, 4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 5376])
        h_v_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_v_fc1) + b_v_fc1)

        h_o_fc1 = tf.nn.relu(tf.matmul(objectInput_flat, W_o_fc1) + b_o_fc1)
        h_o_fc2 = tf.nn.relu(tf.matmul(h_o_fc1, W_o_fc2) + b_o_fc2)

        fc_01 = tf.reshape(tf.concat([h_v_fc1, h_o_fc2], axis=1), [-1, 1536])
        fc_0 = tf.reshape(fc_01, [batch_size, trainLength, 1536])
        fc_0 = tf.nn.dropout(fc_0, keep_prob)

        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True) for size in [1536]*self.config["RNN_LAYERS"]]
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        rnn_state_in = rnn_cell.zero_state(batch_size, tf.float32)
        rnn, rnn_state_out = tf.nn.dynamic_rnn(
            inputs=fc_0, cell=rnn_cell, dtype="float", initial_state=rnn_state_in, scope=my_scope+'_rnn')
        rnn = tf.reshape(rnn, shape=[-1, 1536])

        h_fc1 = tf.nn.relu(tf.matmul(rnn, W_fc1) + b_fc1)

        Adv_split, Val_split = tf.split(h_fc1, 2, 1)

        Adv_split = tf.reshape(tf.concat([Adv_split, allowedActionsInput], axis=1), [-1, 512 + self.actions])
        Advantage = tf.matmul(Adv_split, W_fc_Adv) + b_fc_Adv
        Value = tf.matmul(Val_split, W_fc_Val) + b_fc_Val

        QValue = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keep_dims=True))

        return [viewInput, objectInput, rnn_state_in, rnn_state_out, trainLength, batch_size, allowedActionsInput,
                keep_prob], QValue, None, [W_conv0, b_conv0, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3,
               W_v_fc1, b_v_fc1, W_o_fc1, b_o_fc1, W_o_fc2, b_o_fc2, W_fc1, b_fc1, W_fc_Val, b_fc_Val, W_fc_Adv, b_fc_Adv]

    def trainDRQNNetwork(self, minibatch):
        view_batch = [exp[0][0] for ep in minibatch for exp in ep]
        objects_batch = [exp[0][1] for ep in minibatch for exp in ep]
        allowedActions_batch = [exp[0][2] for ep in minibatch for exp in ep]
        action_batch = [exp[1] for ep in minibatch for exp in ep]
        reward_batch = [exp[2] for ep in minibatch for exp in ep]
        nextView_batch = [exp[3][0] for ep in minibatch for exp in ep]
        nextObjects_batch = [exp[3][1] for ep in minibatch for exp in ep]
        nextAllowedActions_batch = [exp[3][2] for ep in minibatch for exp in ep]
        terminal = [exp[4] for ep in minibatch for exp in ep]

        state_train = tuple([np.zeros([self.config["BATCH_SIZE"], 1536]) for _ in range(self.config["RNN_LAYERS"] * 2)])

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={
            self.inputs_outputsT[0]: nextView_batch,
            self.inputs_outputsT[1]: nextObjects_batch,
            self.inputs_outputsT[2]: state_train,
            self.inputs_outputsT[4]: self.train_length,
            self.inputs_outputsT[5]: self.config["BATCH_SIZE"],
            self.inputs_outputsT[6]: nextAllowedActions_batch,
            self.inputs_outputsT[7]: 1.0,
        })
        for i in range(0, self.config["BATCH_SIZE"] * self.train_length):
            if terminal[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.config["GAMMA"] * np.max(QValue_batch[i]))

        self.merged_summary = tf.summary.merge(self.train_summaries)
        _, summary = self.session.run([self.train_dqn, self.merged_summary], feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.inputs_outputs[0]: view_batch,
            self.inputs_outputs[1]: objects_batch,
            self.inputs_outputs[2]: state_train,
            self.inputs_outputs[4]: self.train_length,
            self.inputs_outputs[5]: self.config["BATCH_SIZE"],
            self.inputs_outputs[6]: allowedActions_batch,
            self.inputs_outputs[7]: self.config["element_wise_keep_prob"],
        })
        return summary

    def getDRQNAction(self):
        QValue, self.current_rnn_state_1 = self.session.run([self.QValue, self.inputs_outputs[3]], feed_dict={
            self.inputs_outputs[0]: [self.current_state[0]],
            self.inputs_outputs[1]: [self.current_state[1]],
            self.inputs_outputs[2]: self.current_rnn_state_1,
            self.inputs_outputs[4]: 1,
            self.inputs_outputs[5]: 1,
            self.inputs_outputs[6]: [self.current_state[2]],
            self.inputs_outputs[7]: 1.0,
        })
        QValue = QValue[0]
        action = np.zeros(self.actions)
        if self.time_step % self.config["FRAME_PER_ACTION"] == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 0

        return action, self.softmax(QValue).tolist()

    def createDQNNetwork(self):
        # network weights / conv filters
        # width of the kernel x height of the kernel x depth of the kernel (input channels) x output dimension (output channels)
        # this kernel fits image width - kernel width + 1 times on the the image (VALID padding)
        W_conv0 = self.weight_variable([10, 12, 3, 16])
        b_conv0 = self.bias_variable([16])

        W_conv1 = self.weight_variable([7, 8, 16, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([6, 6, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_v_fc1 = self.weight_variable([5376, 495])
        b_v_fc1 = self.bias_variable([495])

        #######

        W_o_fc1 = self.weight_variable([495, 990])
        b_o_fc1 = self.bias_variable([990])

        W_o_fc2 = self.weight_variable([990, 495])
        b_o_fc2 = self.bias_variable([495])

        #######

        W_fc1 = self.weight_variable([990, 1980])
        b_fc1 = self.bias_variable([1980])

        W_fc2 = self.weight_variable([1980, 1980])
        b_fc2 = self.bias_variable([1980])

        W_fc3 = self.weight_variable([1980, 1980])
        b_fc3 = self.bias_variable([1980])

        W_fc4 = self.weight_variable([1980, 990])
        b_fc4 = self.bias_variable([990])

        W_fcx = self.weight_variable([990 + self.actions, self.actions])
        b_fcx = self.bias_variable([self.actions])

        W_fc_Val = self.weight_variable([512, 1])
        b_fc_Val = self.bias_variable([1])

        W_fc_Adv = self.weight_variable([512 + self.actions, self.actions])
        b_fc_Adv = self.bias_variable([self.actions])


        keep_prob = tf.placeholder(tf.float32)

        # input layer
        # batch size x image height x image width x 4 consecutive images (channels)
        viewInput = tf.placeholder("float", [None, 338, 600, 3])
        objectInput = tf.placeholder("float", [None, 33, 5, 3])
        objectInput_flat = tf.reshape(objectInput, [-1, 495])
        allowedActionsInput = tf.placeholder("float", [None, self.actions])

        # hidden layers
        # we obtain a 77x77 output with depth of 32 from conv2d without stride
        # with stride of 4: 20x20 with depth 32 -> [1,20,20,32]
        # all values in this matrix correspond to output values, flattened we would have 12800 values
        # test = self.conv2d(stateInput, W_conv1, 4) + b_conv1
        ##shape = test.get_shape().as_list()
        ##print("dimension:", shape[0], shape[1], shape[2], shape[3])
        h_conv0 = tf.nn.relu(self.conv2d(viewInput, W_conv0, 4) + b_conv0)

        h_conv1 = tf.nn.relu(self.conv2d(h_conv0, W_conv1, 4) + b_conv1)
        # h_pool1 = self.max_pool_2x2(h_conv1)

        # we get [1,9,9,64]
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)

        # [1,7,7,64]
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        # h_conv3_shape = h_conv3.get_shape().as_list()
        # print("dimension:", h_conv3_shape[0], h_conv3_shape[1], h_conv3_shape[2], h_conv3_shape[3])
        # here we get one row for each image [None, 3136]
        h_conv3_flat = tf.reshape(h_conv3, [-1, 5376])
        # h_conv3_shape = h_conv3_flat.get_shape().as_list()
        # print("dimension2:", h_conv3_shape[0], h_conv3_shape[1])
        # [None, 512]
        h_v_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_v_fc1) + b_v_fc1)

        h_o_fc1 = tf.nn.relu(tf.matmul(objectInput_flat, W_o_fc1) + b_o_fc1)
        h_o_fc2 = tf.nn.relu(tf.matmul(h_o_fc1, W_o_fc2) + b_o_fc2)

        fc_0 = tf.reshape(tf.stack([h_v_fc1, h_o_fc2], axis=2), [-1, 990])
        fc_0 = tf.nn.dropout(fc_0, keep_prob)

        h_fc1 = tf.nn.relu(tf.matmul(fc_0, W_fc1) + b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

        h_fc4 = tf.reshape(tf.concat([h_fc4, allowedActionsInput], axis=1), [-1, 990 + self.actions])

        # Q Value layer
        # [None, self.actions]
        QValue = tf.matmul(h_fc4, W_fcx) + b_fcx
        # shape = QValue.get_shape().as_list()
        # print("dimension QValue:", shape[0], shape[1])

        return [viewInput, objectInput, allowedActionsInput, keep_prob], QValue, None, [W_conv0, b_conv0, W_conv1, b_conv1,
                W_conv2, b_conv2, W_conv3, b_conv3, W_v_fc1, b_v_fc1, W_o_fc1, b_o_fc1, W_o_fc2, b_o_fc2, W_fc1, b_fc1,
                W_fc2, b_fc2, W_fc3, b_fc3, W_fc4, b_fc4, W_fcx, b_fcx]

    def trainDQNNetwork(self, minibatch):
        view_batch = [exp[0][0] for exp in minibatch]
        objects_batch = [exp[0][1] for exp in minibatch]
        allowedActions_batch = [exp[0][2] for exp in minibatch]
        action_batch = [exp[1] for exp in minibatch]
        reward_batch = [exp[2] for exp in minibatch]
        nextView_batch = [exp[3][0] for exp in minibatch]
        nextObjects_batch = [exp[3][1] for exp in minibatch]
        nextAllowedActions_batch = [exp[3][2] for exp in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={
            self.inputs_outputsT[0]: nextView_batch,
            self.inputs_outputsT[1]: nextObjects_batch,
            self.inputs_outputsT[2]: nextAllowedActions_batch,
            self.inputs_outputsT[3]: 1.0,
        })
        for i in range(0, self.config["BATCH_SIZE"]):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.config["GAMMA"] * np.max(QValue_batch[i]))

        _, summary = self.session.run([self.train_dqn, self.merged_summary], feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.inputs_outputs[0]: view_batch,
            self.inputs_outputs[1]: objects_batch,
            self.inputs_outputs[2]: allowedActions_batch,
            self.inputs_outputs[3]: self.config["element_wise_keep_prob"],
        })
        return summary

    def getDQNAction(self):
        QValue = self.session.run([self.QValue], feed_dict={
            self.inputs_outputs[0]: [self.current_state[0]],
            self.inputs_outputs[1]: [self.current_state[1]],
            self.inputs_outputs[2]: [self.current_state[2]],
            self.inputs_outputs[3]: 1.0,
        })
        QValue = QValue[0]
        action = np.zeros(self.actions)
        if self.time_step % self.config["FRAME_PER_ACTION"] == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 0

        return action, self.softmax(QValue).tolist()



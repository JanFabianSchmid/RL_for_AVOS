from config import config
from ActiveVisualObjectSearch import ActiveVisualObjectSearchEnv
import sys
import time
import os
from os import listdir
from Brain import *

WIDTH = config["image_size"][0]
HEIGHT = config["image_size"][1]
COLORS = config["image_size"][2]

start = time.time()

postfix = ''
if len(sys.argv) > 1:
    postfix = sys.argv[1]

parameter_identifier = 'default'
agent = 'DRQN'
object_proposal_detector = 'faster_rcnn'
start_with_positive_samples = False
test_on_train_set = False
test_mode = False
test_series = False
test_series_file = 'test_setups/exploration_familiar_objects_familiar_scenes'
familiar_objects = True
familiar_scenes = True
view_image_input = True
object_detector_input = True
cropped_object_proposal_input = True
allowed_actions_input = True
use_bottlenecks = True
computing_on_cluster = True
bayesian_movement = False
bayesian_proposal_classifier = False
bayesian_cheating = False
logging = False

for idx in range(len(sys.argv)-2):
    if sys.argv[idx+2] == 'dqn':
        agent = 'DQN'  # default, with_allowed_actions, without_view_image, only_view_image
    elif sys.argv[idx+2] == 'drqn':
        agent = 'DRQN'
    elif sys.argv[idx+2] == 'drascos':
        agent = 'DRASCOS'
    elif sys.argv[idx+2] == 'drascos_no_rnn':
        agent = 'DRASCOS_no_rnn'
    elif sys.argv[idx+2] == 'bayesian':
        agent = 'bayesian'
    elif sys.argv[idx+2] == 'bayesian_with':
        agent = 'bayesian_with_classifier'
    elif sys.argv[idx+2] == 'random_with':
        agent = 'random_with_classifier'
    elif sys.argv[idx + 2] == 'rotating_with':
        agent = 'rotating_with_classifier'
    elif sys.argv[idx+2] == 'drqn_with':
        agent = 'DRQN_with_auxiliary'
    elif sys.argv[idx + 2] == 'drqn_prop':
        agent = 'DRQN_with_prop'
    elif sys.argv[idx + 2] == 'drqn_scene':
        agent = 'DRQN_with_scene'
    elif sys.argv[idx + 2] == 'drqn_prop_small':
        agent = 'DRQN_with_prop_small'
    elif sys.argv[idx+2] == 'random':
        agent = 'random'
    elif sys.argv[idx+2] == 'avd':
        agent = 'avd_baseline'
    elif sys.argv[idx+2] == 'gt':
        object_proposal_detector = 'gt'
    elif sys.argv[idx + 2] == 'faster_rcnn':
        object_proposal_detector = 'faster_rcnn'
    elif sys.argv[idx+2] == 'start_without':
        start_with_positive_samples = False
    elif sys.argv[idx+2] == 'start_with':
        start_with_positive_samples = True
    elif sys.argv[idx+2] == 'test_on_train':
        #test_on_train_set = True
        test_series = True
        test_series_file = 'test_setups/training_set'
        familiar_objects = True
        familiar_scenes = True
    elif sys.argv[idx+2] == 'test':
        test_mode = True
    elif sys.argv[idx+2] == 'test_series':
        test_series = True
    elif sys.argv[idx + 2] == 'test_on_familiar_scenes_with_familiar_objects':
        test_series = True
        test_series_file = 'test_setups/exploration_familiar_objects_familiar_scenes'
        familiar_objects = True
        familiar_scenes = True
    elif sys.argv[idx + 2] == 'test_on_unfamiliar_scenes_with_familiar_objects':
        test_series = True
        test_series_file = 'test_setups/exploration_familiar_objects_unfamiliar_scenes'
        familiar_objects = True
        familiar_scenes = False
    elif sys.argv[idx + 2] == 'test_on_familiar_scenes_with_unfamiliar_objects':
        test_series = True
        test_series_file = 'test_setups/exploration_unfamiliar_objects_familiar_scenes'
        familiar_objects = False
        familiar_scenes = True
    elif sys.argv[idx+2] == 'without_image':
        view_image_input = False
        parameter_identifier = 'without_image'
    elif sys.argv[idx+2] == 'without_object_detector':
        object_detector_input = False
        parameter_identifier = 'without_object_detector'
    elif sys.argv[idx+2] == 'without_cropped_proposals':
        cropped_object_proposal_input = False
        parameter_identifier = 'without_cropped_proposals'
    elif sys.argv[idx+2] == 'without_actions':
        allowed_actions_input = False
    elif sys.argv[idx + 2] == 'full_image':
        use_bottlenecks = False
    elif sys.argv[idx + 2] == 'bottleneck':
        use_bottlenecks = True
    elif sys.argv[idx + 2] == 'home_pc':
        computing_on_cluster = False
    elif sys.argv[idx + 2] == 'cluster':
        computing_on_cluster = True
    elif sys.argv[idx + 2] == 'bayesian_movement':
        bayesian_movement = True
    elif sys.argv[idx + 2] == 'bayesian_proposals':
        bayesian_proposal_classifier = True
    elif sys.argv[idx + 2] == 'cheating':
        bayesian_cheating = True
        parameter_identifier = 'cheating'
    elif sys.argv[idx + 2] == 'log':
        logging = True

if agent == 'bayesian' or agent == 'bayesian_with_classifier':
    bayesian_movement = True
    bayesian_proposal_classifier = True

print("Running agent with following parameters:")
print("  File postfix: ", postfix)
print("  Agent: ", agent)
print("  Object proposal detector: ", object_proposal_detector)
print("  Starting with pre-recorded positive episodes: ", start_with_positive_samples)
print("  Testing is done on training data: ", test_on_train_set)
print("  Only testing: ", test_mode)
print("  Using bottlenecks instead of full images: ", use_bottlenecks)
print("  Computing is performed on cluster: ", computing_on_cluster)
print("Inputs used:")
print("  View image: ", view_image_input)
print("  Cropped object proposal: ", cropped_object_proposal_input)
print("  Object detector: ", object_detector_input)
print("  Available actions: ", allowed_actions_input)
print("Usage of Bayesian reasoning")
print("  Greedy Bayesian movement policy", bayesian_movement)
print("  Bayesian proposal classifier", bayesian_proposal_classifier)
print("  Bayesian cheating", bayesian_cheating)

def add_noise(image, mean, std):
    noisy = image + np.random.normal(mean, std, image.shape)
    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype(np.uint8)
    return noisy

def pre_process(observation):
    if not view_image_input:
        if not use_bottlenecks:
            observation[0] = np.zeros(shape=observation[0].shape).astype(int)
        else:
            observation[0] = np.zeros(shape=observation[0].shape)
    #if not object_detector_input: # this removes not only the classifier certainty
    #    observation[1] = np.zeros(shape=observation[1].shape)
    #    print("obs:", str(observation[1]))
    if not allowed_actions_input:
        observation[2] = [0] * len(observation[2])
    if not cropped_object_proposal_input:
        observation[3] = np.zeros(shape=observation[3].shape)

    #print(np.round(observation[0], decimals=3))

    return observation

def get_label(env):
    label = np.zeros(2)
    if env.should_have_proposed:
        label[0] = 1
    else:
        label[1] = 1
    return label

class positive_sample_generator():
    def __init__(self, data_path):
        self.data_path = data_path
        self.sample_count = 0
        self.log_list = listdir(data_path)
        self.current_sample = None
        self.current_log_path = None
        self.current_action = None

    def prepare_env(self, env):
        self.current_log_path = os.path.join(self.data_path, self.log_list[self.sample_count])
        with open(self.current_log_path+'/stats', mode='r') as file:
            stats = []
            for line in file:
                stats = line.split(',')
            env.set_scene_name(stats[3])
            env.set_target_object_name(stats[4])
            env.set_first_image(stats[5])
        self.sample_count += 1
        self.current_action = 0
        self.current_sample = self.get_positive_episode()
        if self.current_sample is None:
            return False
        return True

    def get_positive_episode(self):
        if len(self.log_list) <= self.sample_count:
            return None
        first = True
        action_list = []
        with open(self.current_log_path+'/log', mode='r') as file:
            for line in file:
                if first:
                    first = False
                else:
                    line = line.strip()
                    step = line.split(';')
                    action = step[0][1:-1].split(',')
                    action[0] = int(action[0])
                    action[1] = int(action[1])
                    action[2] = int(action[2])
                    action_list.append(action)
        return action_list

    def getAction(self):
        env_action = self.current_sample[self.current_action]
        brain_action = [0, 0, 0, 0, 0, 0, 0]
        if env_action[2] == 1:
            brain_action[0] = 1
        elif env_action[0] == 1:
            brain_action[1] = 1
        elif env_action[0] == 2:
            brain_action[2] = 1
        elif env_action[0] == 3:
            brain_action[3] = 1
        elif env_action[0] == 4:
            brain_action[4] = 1
        elif env_action[1] == 1:
            brain_action[5] = 1
        elif env_action[1] == 2:
            brain_action[6] = 1
        self.current_action += 1
        return brain_action, [0, 0, 0, 0, 0, 0, 0]

def sample_random_action():
    moves = ['propose', 'forward', 'backward', 'left', 'right', 'rotate_ccw', 'rotate_cw']
    action = [0, 0, 0]
    actionmax = -1
    action_tested = []
    while not len(action_tested) == 6:
        test_actionmax = random.randint(0, 6)  ### 5 if no proposal
        if test_actionmax not in action_tested:
            action_tested.append(test_actionmax)
        if test_actionmax == 0:  ###
            actionmax = 0  ###
            break  ###
        pose = env.annotations[env.cur_image_name][moves[test_actionmax]]
        if not env.get_move_image(moves[test_actionmax]) == '':
            actionmax = test_actionmax
            if not env.pose_was_already_visited(pose):
                break

    # actionmax += 1   ###
    if actionmax == 0:  # make proposal
        action[2] = 1
    elif actionmax == 1:  # move forwards
        action[0] = 1
    elif actionmax == 2:  # move backwards
        action[0] = 2
    elif actionmax == 3:  # move left
        action[0] = 3
    elif actionmax == 4:  # move right
        action[0] = 4
    elif actionmax == 5:  # rotate left
        action[1] = 1
    elif actionmax == 6:  # rotate right
        action[1] = 2
    return action

env = ActiveVisualObjectSearchEnv()
env.set_agent(agent)
env.set_parameter_identifier(parameter_identifier)
env.set_postfix(postfix)
env.set_object_detector(object_proposal_detector)
env.set_test_on_train_set(test_on_train_set)
env.set_use_bottlenecks(use_bottlenecks)
env.set_familiar_objects(familiar_objects)
env.set_familiar_scenes(familiar_scenes)
env.set_logging(logging)

#scene = 'Home_001_2'
#object = 'aunt_jemima_original_syrup'
#image = '000120010840101.jpg'

#env.set_scene_name(scene)
#env.set_target_object_name(object)
#env.set_first_image(image)

actions = [0] * 7
brain_agent_type = agent
if agent == 'bayesian_with_classifier' or agent == 'bayesian' or agent == 'avd_baseline' or agent == 'random' or agent == 'rotating_with_classifier':
    brain_agent_type = 'random_with_classifier'
brain = Network(brain_agent_type, parameter_identifier, len(actions), postfix, use_bottlenecks, computing_on_cluster)

if start_with_positive_samples:
    psg = positive_sample_generator('positive_logs')
    start_with_positive_samples = psg.prepare_env(env)

if test_mode or test_series:
    env.set_test_modus(True)
    brain.set_test_modus(True)
    brain.epsilon = 0.0
    if test_series:
        env.start_test_series(test_series_file)
        test_series = env.test_series_still_going()

if not object_detector_input:
    env.remove_classifier_certainty = True

observation0 = pre_process(env.reset())
brain.setInitState(observation0)

statsComputer = None
unexplored_voxels = 0; unexplored_voxel_prob = 0; summed_probs_overall = 0; prev_move = None; camera_info = None
if bayesian_movement or bayesian_proposal_classifier:
    import data_publisher_bridge as dpb
    statsComputer = dpb.StatsComputer()
    gw = dpb.generate_gateway()
    unexplored_voxels, unexplored_voxel_prob, summed_probs_overall, prev_move, camera_info = dpb.reset_octomap(env, gw)

game_count = env.game_count
won = 0
won_in_20 = 0
won_in_100 = 0
backupepsilon = brain.epsilon
won_this_game = False
terminal = False
make_hypothesis = False

while True:
    action = [0, 0, 0]
    QValue = None
    actionmax = -1

    if agent == 'random':
        action = sample_random_action()
    elif agent == 'avd_baseline':
        action, certainty = env.get_AVD_baseline_action()
        if certainty > 0.5:  # make proposal
            action = [0, 0, 1]
        if certainty < 0.1:  # random movement
            found_move = False
            while not found_move:
                action = sample_random_action()
                if action[2] == 0:
                    found_move = True
    else:
        if not (bayesian_movement and bayesian_proposal_classifier and not agent == 'bayesian_with_classifier'):
            if start_with_positive_samples:
                brain_action, QValue = psg.getAction()
            if not start_with_positive_samples:
                brain_action, QValue = brain.getAction()
            actionmax = np.argmax(np.array(brain_action))

    if bayesian_movement or bayesian_proposal_classifier:
        use_proposal = None
        FNR = None
        FPR = None
        if agent == 'bayesian_with_classifier':
            use_proposal = actionmax == 0
            FNR = 0.7
            FPR = 0.01
        unexplored_voxels, unexplored_voxel_prob, summed_probs_overall, make_hypothesis = dpb.process_observation(env, gw,
            statsComputer, unexplored_voxels, unexplored_voxel_prob, summed_probs_overall, use_proposal, FNR, FPR, camera_info, bayesian_cheating)
        if make_hypothesis:
            QValue = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if not agent == 'avd_baseline' and not agent == 'random':
        if (not bayesian_proposal_classifier and actionmax == 0) or (bayesian_proposal_classifier and make_hypothesis):  # make proposal
            action[2] = 1
        elif bayesian_movement:
            action, prev_move, QValue = dpb.find_best_move(env, gw, unexplored_voxels, unexplored_voxel_prob, summed_probs_overall, prev_move, camera_info, bayesian_cheating)
            print(QValue)
        elif actionmax == 1:  # move forwards
            action[0] = 1
        elif actionmax == 2:  # move backwards
            action[0] = 2
        elif actionmax == 3:  # move left
            action[0] = 3
        elif actionmax == 4:  # move right
            action[0] = 4
        elif actionmax == 5:  # rotate left
            action[1] = 1
        elif actionmax == 6:  # rotate right
            action[1] = 2
        else:
            print("Something went wrong with the number of actions")

    if agent == 'rotating_with_classifier' and action[2] == 0:
        action = [0, 1, 0]

    # label for previous observation is stored before we get the next observation
    label = get_label(env)

    env.setQValue(QValue)
    nextObservation, reward, terminal, info = env.step(action)
    #brain.log_scalar('time_per_action', time.time() - start)
    start = time.time()
    nextObservation = pre_process(nextObservation)
    #end = time.time()
    #print("Pre-process time: ", end - start)
    if info['success']:
        if won_this_game:
            print("WON MULTIPLE TIMES IN ONE GAME!")
        won_this_game = True
        if not test_series:
            won += 1
            won_in_20 += 1
            won_in_100 += 1

    if not terminal and action[2] == 1:
        terminal = True  # this agent cannot continue its search once a proposal was made
        env.write_out_statistics()

    if not agent == 'avd_baseline' and not agent == 'random':
        if not (bayesian_movement and bayesian_proposal_classifier and not agent == 'bayesian_with_classifier'):
            brain.setPerception(nextObservation, brain_action, reward, terminal, label, env.get_target_scene_vector())

    if terminal:
        #print(str(won_this_game) + ", " + str(info['total_reward']))
        #brain.log_scalar('total_reward', info['total_reward'])
        test_series = env.test_series_still_going()
        if not test_series:
            game_count += 1
            if (game_count) % 20 == 0:
                print("Successrate in the last 20 games:", won_in_20/20)
                won_in_20 = 0
                print("Starting game: ", game_count)
            if (game_count) % 100 == 0:
                print("Successrate in the last 100 games:", won_in_100/100)
                #brain.log_scalar('successrate_in_100', won_in_100/100)
                won_in_100 = 0
            if ((game_count) % 20 == 0 or test_mode) and not start_with_positive_samples:
                env.set_test_modus(True)
                brain.set_test_modus(True)
                backupepsilon = brain.epsilon
                brain.epsilon = 0.0
            else:
                env.set_test_modus(False)
                brain.set_test_modus(False)
                brain.epsilon = backupepsilon
        env.unset_scenario()
        #env.set_scene_name(scene)
        #env.set_target_object_name(object)
        #env.set_first_image(image)
        if start_with_positive_samples and not test_series:
            start_with_positive_samples = psg.prepare_env(env)
        nextObservation = pre_process(env.reset())
        won_this_game = False
        brain.setInitState(nextObservation)
        if bayesian_movement or bayesian_proposal_classifier:
            unexplored_voxels, unexplored_voxel_prob, summed_probs_overall, prev_move, camera_info = dpb.reset_octomap(env, gw)



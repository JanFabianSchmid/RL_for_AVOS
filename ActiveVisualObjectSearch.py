from config import config
from config import instance_names
from config import nature_valley
from config import soap
from config import honey_bunches
from config import scene_list

import os
import os.path
import errno
import json
import math
from scipy import misc
import scipy.io as sio
from random import randint
import random
import numpy as np
from collections import deque

RENDER_MODE = config["render_mode"]
WIDTH = config["image_size"][0]
HEIGHT = config["image_size"][1]
COLORS = config["image_size"][2]
MAX_STEPS = config["max_steps"]


class ActiveVisualObjectSearchEnv:
    def __init__(self):
        self.data_path = 'ActiveVisionDataset'
        self.test_modus = False
        self.scene_name = None
        self.scene_has_been_set = False
        self.target_object_name = None
        self.target_object_has_been_set = False
        self.first_image = None
        self.first_image_has_been_set = False
        self.parameter_identifier = "default"
        self.object_proposal_detector = 'gt'
        self.scale_factor = config["image_size"][0]/config["original_image_size"][0]
        self.nof_objects = len(instance_names) - 1
        self.info = {}
        self.info["success"] = False
        self.info["total_reward"] = 0
        self.previous_certainty = 0.0
        self.target_available = False
        self.images_path = None
        self.postfix = ''
        self.target_object_id = None
        self.best_matching_object = None
        self.ids = None
        self.QValue = None
        self.should_have_proposed = False
        self.target_boundary_defined = False
        self.test_on_train_set = False
        self.use_bottlenecks = False
        self.target_object_idx = None
        self.bottleneck_creator = None
        self.mirrored = False
        self.set_of_visited_poses = set()
        self.logging = False
        self.test_series = False
        self.test_series_game_count = 0
        self.test_series_setups = []
        self.test_series_length = 0
        self.test_series_count = 0
        self.test_series_success_count = 0
        self.test_series_steps = 0
        self.test_series_object = None
        self.game_count = 0
        self.current_certainty = 0
        self.first_target_object_encounter = None
        self.initial_certainty = 0.0
        self.certainty_change = []
        self.distance_change = []
        self.initial_distance = 0.0
        self.current_distance = 0.0
        self.previous_distance = 0.0
        self.image_positions = {}
        self.object_locations = {}
        self.remove_classifier_certainty = False
        self.familiar_scenes = True
        self.familiar_objects = True
        self.test_object_list = list(config["test_object_list_familiar_objects"])
        self.test_scene_list = list(config["test_scene_list_familiar_scenes"])


    def set_object_detector(self, object_detector):
        if object_detector in ['gt', 'faster_rcnn']:
            self.object_proposal_detector = object_detector
        else:
            print("Please select a valid object detector: gt, faster_rcnn")

    def set_test_on_train_set(self, test_on_train_set):
        self.test_on_train_set = test_on_train_set

    def set_test_modus(self, test_modus):
        self.test_modus = test_modus

    def set_agent(self, agent):
        self.agent = agent

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_familiar_scenes(self, familiar_scenes):
        self.familiar_scenes = familiar_scenes
        if familiar_scenes:
            self.test_scene_list = config["test_scene_list_familiar_scenes"]
        else:
            self.test_scene_list = config["test_scene_list_unfamiliar_scenes"]

    def set_familiar_objects(self, familiar_objects):
        self.familiar_objects = familiar_objects
        if familiar_objects:
            self.test_object_list = config["test_object_list_familiar_objects"]
        else:
            self.test_object_list = config["test_object_list_unfamiliar_objects"]

    def start_test_series(self, file_name):
        self.test_series = True
        self.test_series_game_count = 0
        self.test_series_setups = []
        self.test_series_length = 0
        self.test_series_count = self.game_count
        self.test_series_success_count = 0
        self.test_series_steps = 0
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                self.test_series_setups.append(line.split(','))
                self.test_series_length += 1

    def test_series_still_going(self):
        return self.test_series and self.test_series_game_count < self.test_series_length

    def next_test_series_setup(self):
        print("Next test setup: " + str(self.test_series_setups[self.test_series_game_count]))
        self.set_scene_name(self.test_series_setups[self.test_series_game_count][0])
        target_object = self.test_series_setups[self.test_series_game_count][1]
        self.test_series_object = target_object
        if self.test_series_game_count % 2 == 0 and config['perfect_approaching_proposal_test']:
            target_object = self.test_object_list[int(self.test_series_game_count/2) % len(self.test_object_list)]
            if target_object == self.test_series_object:
                target_object = self.test_object_list[(int(self.test_series_game_count/2) + 1) % len(self.test_object_list)]

        self.set_target_object_name(target_object)
        self.set_first_image(self.test_series_setups[self.test_series_game_count][2])

    def set_logging(self, logging):
        self.logging = logging

    def set_cropped_object_proposal_input(self, cropped_object_proposal_input):
        if cropped_object_proposal_input:
            self.cropped_object_proposal_input = True
            import active_vision.generate_bottleneck_files as gbf
            self.bottleneck_creator = gbf.bottleneck_creator()

    def set_scene_name(self, scene_name):
        """
        Sets the scene used for the next round
        :param scene_name:
        """
        self.scene_name = scene_name
        self.scene_has_been_set = True

    def set_target_object_name(self, target_object_name):
        self.target_object_name = str(target_object_name)
        self.target_object_has_been_set = True

    def set_first_image(self, first_image):
        self.first_image = first_image
        self.first_image_has_been_set = True

    def set_use_bottlenecks(self, use_bottlenecks):
        self.use_bottlenecks = use_bottlenecks

    def unset_scenario(self):
        self.scene_has_been_set = False
        self.target_object_has_been_set = False
        self.first_image_has_been_set = False

    def set_parameter_identifier(self, parameter_identifier):
        self.parameter_identifier = str(parameter_identifier)

    def set_game_count(self, game_count):
        self.game_count = game_count

    def set_postfix(self, postfix):
        self.postfix = postfix

    def _create_logging_files(self):
        if self.test_series and self.test_series_game_count == 0:
            #stats_file_exists = True
            #while stats_file_exists:
            #    stats_file_exists = os.path.isfile(
            #        config["statistics_file_path"] + '_test_series_' + str(self.test_series_count) + self.postfix)
            #    if stats_file_exists:
            #        self.test_series_count += 1
            if not os.path.isfile('test_series_results_' + str(self.test_series_count) + self.postfix):
                with open('test_series_results_' + str(self.test_series_count) + self.postfix,
                          'w') as file:
                    file.write("agent,configuration,object_proposal_detector,scene,object,start_image,test_modus,success,steps,"
                               "total_reward,game_count,distanceToTargetVisibility,first_target_object_encounter,initial_certainty,initial_distance")
                    for idx in range(10):
                        file.write(',certainty_change' + str(idx+1))
                        file.write(',distance_change' + str(idx + 1))
                    file.write('\n')
        else:
            if not os.path.isfile(config["statistics_file_path"] + self.postfix):
                with open(config["statistics_file_path"] + self.postfix, 'w') as file:
                    file.write("agent,configuration,object_proposal_detector,scene,object,start_image,test_modus,success,steps,"
                               "total_reward,game_count,distanceToTargetVisibility,first_target_object_encounter,initial_certainty,initial_distance")
                    for idx in range(10):
                        file.write(',certainty_change' + str(idx+1))
                        file.write(',distance_change' + str(idx + 1))
                    file.write('\n')

        if self.logging:
            try:
                os.makedirs(config["log_folder_path"] + self.postfix)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            log_folder_exists = True
            self.experiment_count = 0
            while log_folder_exists:
                log_folder_exists = os.path.isdir(config["log_folder_path"] + self.postfix + "/log_" + str(self.experiment_count))
                if log_folder_exists:
                    self.experiment_count += 1
            try:
                os.makedirs(config["log_folder_path"] + self.postfix + "/log_" + str(self.experiment_count))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            with open(config["log_folder_path"] + self.postfix + "/log_" + str(self.experiment_count) + "/log", 'w') as file:
                file.write("action;IoU;reward;pred_boundary_box;certainty;gt_boundary_box;QValues;distanceToTargetVisibility\n")

    def reset(self):
        if self.game_count == 0:
            try:
                with open(config["statistics_file_path"] + self.postfix, 'r') as file:
                    #print("Old session is continued")
                    number_of_lines = 0
                    for line in file:
                        number_of_lines += 1
                        line = line.split(',')
                    if number_of_lines > 1:
                        self.game_count = int(line[10])
            except OSError:
                print("New session started")

        if not self.scene_has_been_set:
            self.scene_name = None
        if not self.target_object_has_been_set:
            self.target_object_name = None
        if not self.first_image_has_been_set:
            self.first_image = None
        if self.test_series:
            if self.test_series_game_count == 0:
                try:
                    with open('test_series_results_' + str(self.test_series_count) + self.postfix, 'r') as file:
                        number_of_lines = 0
                        for line in file:
                            number_of_lines += 1
                            line = line.split(',')
                        if number_of_lines > 1 and number_of_lines < self.test_series_length:
                            self.test_series_game_count = int(line[10])
                            print("Old session is continued")
                except OSError:
                    print("New test session started")
            self.next_test_series_setup()
        else:
            self.game_count += 1
        self.target_available = False
        self.mirrored = False
        self.set_of_visited_poses.clear()
        self._load_data()

        self.done = False
        self.pred_boundary_box = None
        self.nof_steps = 0
        self.info["success"] = False
        self.info["total_reward"] = 0

        self._create_logging_files()

        _, _, _, certainty, x1, y1, x2, y2 = self.get_object_candidates()
        target_object_vector = self.get_target_object_vector()
        self.target_object_idx = max(enumerate(target_object_vector), key=lambda t: t[1])[0]
        if x1[self.target_object_idx] >= 0:
            proposal_box = [x1[self.target_object_idx], y1[self.target_object_idx], x2[self.target_object_idx], y2[self.target_object_idx]]
            self.should_have_proposed, _, _ = self.check_proposal(proposal_box)

        self.current_certainty = certainty[self.target_object_idx]
        self.initial_certainty = self.current_certainty
        self.previous_certainty = self.current_certainty

        if str(self.target_object_id) in self.object_locations and self.cur_image_name in self.image_positions:
            target_location = self.object_locations[str(self.target_object_id)]
            image_location = self.image_positions[self.cur_image_name]
            self.current_distance = math.sqrt(math.pow(target_location[0] - image_location[0], 2) + math.pow(target_location[2] - image_location[2], 2))
            self.initial_distance = self.current_distance
            self.previous_distance = self.current_distance

        self.certainty_change = []
        self.distance_change = []

        self.initialDistanceToPoseWithTargetVisibility = self.distanceToPoseWithTargetVisibility()
        self.first_target_object_encounter = None
        if self.initialDistanceToPoseWithTargetVisibility == 0:
            self.first_target_object_encounter = 0

        if self.logging:
            if not self.target_boundary_defined:
                if x1[self.target_object_idx] >= 0:
                    proposal_box = [x1[self.target_object_idx], y1[self.target_object_idx], x2[self.target_object_idx],
                                    y2[self.target_object_idx]]
                    self.define_target_boundary(proposal_box, True)
            else:
                self.target_boundary_defined = False
            _, _, gt_boundary_box = self.check_proposal()

            with open(config["log_folder_path"] + self.postfix + "/log_" + str(self.experiment_count) + "/log", 'a') as file:
                file.write(str([0,0,0]) + ';' + str(0.0) + ';' + str(0) + ';' + str(None) + ';' +
                           str(self.current_certainty) + ';' + str(self.current_distance) + ';' + str(gt_boundary_box) + ';' + str(self.QValue) +
                           ';' + str(self.initialDistanceToPoseWithTargetVisibility) + '\n')

        return self.create_observation()

    def _load_data(self):
        modus = 'train'
        scene_list = list(config["train_scene_list"])
        object_list = list(config["train_object_list"])
        if self.test_modus and not self.test_on_train_set:
            modus = 'test'
            scene_list = self.test_scene_list
            object_list = self.test_object_list
        if self.scene_name is None:
            scene_number = randint(0, len(scene_list) - 1)
            self.scene_name = scene_list[scene_number]
            if modus == 'train' and config["use_mirrored_training_scenes"]:
                if random.random() < config["chance_of_using_mirrored_training_scene"]:
                    self.mirrored = True
                    self.scene_name = self.scene_name + '_mirrored'
        present_instance_names_path = os.path.join(self.data_path, self.scene_name, 'present_instance_names.txt')
        present_instance_names = []
        with open(present_instance_names_path, 'r') as f:
            for line in f:
                present_instance_names.append(line.rstrip())
        if self.target_object_name is None:
            if (modus == 'train' and config['use_all_available_objects_in_training']) or \
               (modus == 'test' and config['use_all_available_objects_in_tests']):
                object_list = present_instance_names
            while len(object_list) > 0:
                object_number = randint(0, len(object_list) - 1)
                self.target_object_name = object_list[object_number]
                self.target_object_id = instance_names[self.target_object_name]
                self.target_available = self.target_object_name in present_instance_names
                if self.target_available or config['allow_not_available_target_objects']:
                    break
                else:
                    object_list.remove(self.target_object_name)
        else:
            self.target_object_id = instance_names[self.target_object_name]

        self.best_matching_object = self.target_object_id
        self.ids = [self.target_object_id]
        if self.target_object_id in nature_valley:
            self.ids = nature_valley
        if self.target_object_id in soap:
            self.ids = soap
        if self.target_object_id in honey_bunches:
            self.ids = honey_bunches
        self.target_available = self.target_object_name in present_instance_names
        if not self.target_available and not config['allow_not_available_target_objects']:
            print("ERROR: No possible target object available!")
            print("Scene:", self.scene_name)
            print("Target object:", self.target_object_name)

        if self.agent == 'human':
            print("Scene:", self.scene_name)
            print("Target object:", self.target_object_name)

        # load data
        self.images_path = os.path.join(self.data_path, self.scene_name, 'jpg_rgb')
        annotations_path = os.path.join(self.data_path, self.scene_name, 'annotations_new.json')
        image_names = os.listdir(os.path.join(self.data_path, self.scene_name, 'bottleneck'))
        image_names.sort()
        ann_file = open(annotations_path)
        self.annotations = json.load(ann_file)
        try:
            with open(os.path.join(self.data_path, self.scene_name, 'AVDB', 'object_locations.json'), 'r') as json_file:
                self.object_locations = json.load(json_file)
            with open(os.path.join(self.data_path, self.scene_name, 'AVDB', 'image_positions.json'), 'r') as json_file:
                self.image_positions = json.load(json_file)
        except Exception as _:
            #print("Can not open object locations and image positions")
            #print(os.path.join(self.data_path, self.scene_name, 'AVDB', 'object_locations.json'))
            self.object_locations = {}
            self.image_positions = {}

        if self.first_image is None:
            self.first_image = image_names[randint(0, len(image_names) - 1)][:-4] + ".jpg"
            # set up for first image
            #set = []
            #if self.test_modus:
            #    set_name = 'test_set'
            #else:
            #    set_name = 'training_set'
            #with open(os.path.join(self.data_path, self.scene_name, set_name)) as file:
            #    for line in file:
            #        line = line.strip()
            #        set.append(line)
            #self.first_image = set[randint(0, len(set) - 1)]
        self.cur_image_name = self.first_image
        self.set_of_visited_poses.add(self.cur_image_name)


    def get_move_image(self, move):
        if move == 'forward':
            return self.annotations[self.cur_image_name]['forward']
        if move == 'backward':
            return self.annotations[self.cur_image_name]['backward']
        if move == 'left':
            return self.annotations[self.cur_image_name]['left']
        if move == 'right':
            return self.annotations[self.cur_image_name]['right']
        if move == 'rotate_ccw':
            return self.annotations[self.cur_image_name]['rotate_ccw']
        if move == 'rotate_cw':
            return self.annotations[self.cur_image_name]['rotate_cw']
    def get_forward_image(self):
        return self.annotations[self.cur_image_name]['forward']
    def get_backward_image(self):
        return self.annotations[self.cur_image_name]['backward']
    def get_left_image(self):
        return self.annotations[self.cur_image_name]['left']
    def get_right_image(self):
        return self.annotations[self.cur_image_name]['right']
    def get_rotate_ccw_image(self):
        return self.annotations[self.cur_image_name]['rotate_ccw']
    def get_rotate_cw_image(self):
        return self.annotations[self.cur_image_name]['rotate_cw']

    def step(self, action):
        self.nof_steps += 1
        next_image_name = self.cur_image_name

        if self.test_series and (config['perfect_approaching'] or config['perfect_approaching_proposal_test']) and action[2] == 0:
            best_move, action = self.best_move_towards_object(self.test_series_object)
            print("Performing optimal approaching behavior")

        if action[0] == 1:
            next_image_name = self.annotations[next_image_name]['forward']
            #print("Moving forward")
        elif action[0] == 2:
            next_image_name = self.annotations[next_image_name]['backward']
            #print("Moving backward")
        elif action[0] == 3:
            next_image_name = self.annotations[next_image_name]['left']
            #print("Moving left")
        elif action[0] == 4:
            next_image_name = self.annotations[next_image_name]['right']
            #print("Moving right")
        if next_image_name == '':
            next_image_name = self.cur_image_name
        if action[1] == 1:
            next_image_name = self.annotations[next_image_name]['rotate_ccw']
            #print("Rotating left")
        elif action[1] == 2:
            next_image_name = self.annotations[next_image_name]['rotate_cw']
            #print("Rotating right")

        reward = config["step_punishment"]

        illegal_move = False
        if next_image_name != self.cur_image_name:
            self.cur_image_name = next_image_name
            #if self.pose_was_already_visited(next_image_name):
            #    print("Pose was already visited!")
            self.set_of_visited_poses.add(self.cur_image_name)
        else:
            illegal_move = True

        _, _, _, certainty, x1, y1, x2, y2 = self.get_object_candidates()
        self.current_certainty = certainty[self.target_object_idx]

        if str(self.target_object_id) in self.object_locations and self.cur_image_name in self.image_positions:
            target_location = self.object_locations[str(self.target_object_id)]
            image_location = self.image_positions[self.cur_image_name]
            self.current_distance = math.sqrt(
                math.pow(target_location[0] - image_location[0], 2) + math.pow(target_location[2] - image_location[2], 2))

        if not self.target_boundary_defined:
            if x1[self.target_object_idx] >= 0:
                proposal_box = [x1[self.target_object_idx], y1[self.target_object_idx], x2[self.target_object_idx], y2[self.target_object_idx]]
                self.define_target_boundary(proposal_box, True)
        else:
            self.target_boundary_defined = False
        should_propose, IoU, gt_boundary_box = self.check_proposal()

        if gt_boundary_box is not None and self.first_target_object_encounter is None:
            self.first_target_object_encounter = self.nof_steps

        if not self.should_have_proposed and should_propose:
            reward = config["entering_proposal_position_reward"]
        elif self.should_have_proposed and not should_propose:
            reward = config["leaving_proposal_position_punishment"]
        elif self.should_have_proposed and should_propose:
            reward = config["staying_in_proposal_position_reward"]

        self.should_have_proposed = should_propose

        if illegal_move:
            reward = config["illegal_movement_punishment"]

        #current_certainty = 0
        #for idx in self.ids:
        #    if certainty[idx - 1] > current_certainty:
        #        current_certainty = certainty[idx - 1]
        #if current_certainty > self.previous_certainty + config["increasing_certainty_threshold"]:
        #    reward = config["increasing_certainty_reward"]

        if len(self.certainty_change) < 10:
            self.certainty_change.append(self.current_certainty - self.previous_certainty)
        self.previous_certainty = self.current_certainty

        if len(self.distance_change) < 10:
            self.distance_change.append(self.current_distance - self.previous_distance)
            self.previous_distance = self.current_distance

        if action[2] == 1:
            reward = config["wrong_proposal_punishment"]
            if should_propose:
                self.info["success"] = True
                self.done = True
                reward = config["success_reward"]
            #if self.agent == 'human':
            #    print("IoU:", IoU)

        if self.nof_steps == config["max_steps"]:
            self.done = True

        self.info["total_reward"] += reward

        if self.test_series and config['perfect_approaching_proposal_test'] and action[2] == 0 and best_move is None and self.test_series_game_count % 2 == 0:
            self.info["success"] = True
            self.done = True
        elif self.test_series and config['perfect_approaching_proposal_test'] and action[2] == 0 and best_move is None and self.test_series_game_count % 2 != 0:
            self.done = True

        if self.done:
            self.write_out_statistics()

        if self.logging:
            with open(config["log_folder_path"] + self.postfix + "/log_" + str(self.experiment_count) + "/log", 'a') as file:
                file.write(str(action) + ';' + str(IoU) + ';' + str(reward) + ';' + str(self.pred_boundary_box) + ';' +
                           str(self.current_certainty) + ';' + str(self.current_distance) + ';' + str(gt_boundary_box) + ';' + str(self.QValue) +
                           ';' + str(self.distanceToPoseWithTargetVisibility()) + '\n')

        obs = self.create_observation()
        self.pred_boundary_box = None

        return obs, reward, self.done, self.info

    def check_proposal(self, pred_boundary_box=None, image_name=None):
        if image_name is None:
            image_name = self.cur_image_name
        success = False
        IoU = 0.0
        best_gt_boundary_box = None
        if pred_boundary_box is None:
            pred_boundary_box = self.pred_boundary_box
        for box in self.annotations[image_name]['bounding_boxes']:
            if box[4] in self.ids:
                gt_boundary_box = self.scale_values(box, 4)
                if best_gt_boundary_box is None:
                    best_gt_boundary_box = gt_boundary_box
                if pred_boundary_box is not None:
                    new_IoU = self.calculate_IoU(gt_boundary_box, pred_boundary_box)
                    if new_IoU > IoU:
                        IoU = new_IoU
                        self.best_matching_object = box[4]
                        best_gt_boundary_box = gt_boundary_box
        if IoU > 0.3:
            success = True

        return success, IoU, best_gt_boundary_box

    def setQValue(self, QValue):
        self.QValue = QValue

    def distanceToPoseWithTargetVisibility(self):
        visited_nodes = []
        queue = deque()
        queue.append([self.cur_image_name, 0])
        visited_nodes.append(self.cur_image_name)

        while queue:
            image, distance = queue.popleft()
            _, _, gt_boundary_box = self.check_proposal(image_name=image)
            if gt_boundary_box is not None:
                return distance

            if self.annotations[image]['forward'] not in visited_nodes and self.annotations[image]['forward'] != '':
                queue.append([self.annotations[image]['forward'], distance + 1])
                visited_nodes.append(self.annotations[image]['forward'])
            if self.annotations[image]['backward'] not in visited_nodes and self.annotations[image]['backward'] != '':
                queue.append([self.annotations[image]['backward'], distance + 1])
                visited_nodes.append(self.annotations[image]['backward'])
            if self.annotations[image]['left'] not in visited_nodes and self.annotations[image]['left'] != '':
                queue.append([self.annotations[image]['left'], distance + 1])
                visited_nodes.append(self.annotations[image]['left'])
            if self.annotations[image]['right'] not in visited_nodes and self.annotations[image]['right'] != '':
                queue.append([self.annotations[image]['right'], distance + 1])
                visited_nodes.append(self.annotations[image]['right'])
            if self.annotations[image]['rotate_ccw'] not in visited_nodes and self.annotations[image]['rotate_ccw'] != '':
                queue.append([self.annotations[image]['rotate_ccw'], distance + 1])
                visited_nodes.append(self.annotations[image]['rotate_ccw'])
            if self.annotations[image]['rotate_cw'] not in visited_nodes and self.annotations[image]['rotate_cw'] != '':
                queue.append([self.annotations[image]['rotate_cw'], distance + 1])
                visited_nodes.append(self.annotations[image]['rotate_cw'])

    def write_out_statistics(self):
        stats_path = config["statistics_file_path"] + self.postfix
        game_count = self.game_count
        if self.test_series:
            self.test_series_game_count += 1
            game_count = self.test_series_game_count
            stats_path = 'test_series_results_' + str(self.test_series_count) + self.postfix
            if self.info["success"]:
                self.test_series_success_count += 1
                self.test_series_steps += self.nof_steps
            if self.test_series_game_count == self.test_series_length:
                self.test_series = False
                if self.test_series_success_count == 0:
                    self.test_series_success_count = 1
                with open(stats_path+'_summary', 'w') as file:
                    file.write(str(self.test_series_success_count/self.test_series_game_count) + ', ' + str(self.test_series_steps/self.test_series_success_count))

        with open(stats_path, 'a') as file:
            file.write(str(self.agent) + ',' + self.parameter_identifier + ',' + self.object_proposal_detector + ',' + self.scene_name + ',' +
                       self.target_object_name + ',' + str(self.first_image) + ',' + str(self.test_modus) + ',' +
                       str(self.info["success"]) + ',' + str(self.nof_steps) + ',' + str(
                self.info["total_reward"]) + ',' + str(game_count) + ',' + str(self.initialDistanceToPoseWithTargetVisibility) +
                       ',' + str(self.first_target_object_encounter) + ',' + str(self.initial_certainty) + ',' + str(self.initial_distance))
            for idx in range(10):
                value = 0.0
                if len(self.certainty_change) > idx:
                    value = self.certainty_change[idx]
                file.write(',' + str(value))
                value = 0.0
                if len(self.distance_change) > idx:
                    value = self.distance_change[idx]
                file.write(',' + str(value))
            file.write('\n')

        if self.logging:
            with open(config["log_folder_path"] + self.postfix + "/log_" + str(self.experiment_count) + "/" + config[
                "statistics_file_path"], 'w') as file:
                file.write("agent,configuration,object_proposal_detector,scene,object,start_image,test_modus,success,steps,"
                           "total_reward,game_count,distanceToTargetVisibility,first_target_object_encounter,initial_certainty,initial_distance")
                for idx in range(10):
                    file.write(',certainty_change' + str(idx + 1))
                    file.write(',distance_change' + str(idx + 1))
                file.write('\n')
                file.write(str(self.agent) + ',' + self.parameter_identifier + ',' + self.object_proposal_detector + ',' + self.scene_name + ',' +
                           self.target_object_name + ',' + str(self.first_image) + ',' + str(self.test_modus) + ',' +
                           str(self.info["success"]) + ',' + str(self.nof_steps) + ',' + str(
                    self.info["total_reward"]) + ',' + str(self.game_count) + ',' + str(self.initialDistanceToPoseWithTargetVisibility) +
                           ',' + str(self.first_target_object_encounter) + ',' + str(self.initial_certainty) + ',' + str(self.initial_distance))
                for idx in range(10):
                    value = 0.0
                    if len(self.certainty_change) > idx:
                        value = self.certainty_change[idx]
                    file.write(',' + str(value))
                    value = 0.0
                    if len(self.distance_change) > idx:
                        value = self.distance_change[idx]
                    file.write(',' + str(value))
                file.write('\n')


    def _render(self, mode=RENDER_MODE, close=False):
        return misc.imread(os.path.join(self.images_path, self.cur_image_name))

    def get_target_object_image_section(self, compute_instead_of_load=False):
        _, _, size, _, x1, y1, x2, y2 = self.get_object_candidates(scale=False)
        if size[self.target_object_idx] > 0:
            box = [x1[self.target_object_idx],y1[self.target_object_idx], x2[self.target_object_idx], y2[self.target_object_idx]]
            file_name = self.cur_image_name[:-4]+'_'+str(int(box[0]))+str(int(box[1]))+str(int(box[2]))+str(int(box[3]))+'.jpg'
            cropped_object_proposals_folder = "cropped_object_proposals"
            if self.object_proposal_detector == 'faster_rcnn':
                cropped_object_proposals_folder += "_faster_rcnn"
            image_path = os.path.join(self.data_path, self.scene_name, cropped_object_proposals_folder, file_name)
            if not self.use_bottlenecks:
                return misc.imread(image_path)
            elif compute_instead_of_load and self.bottleneck_creator is not None:
                return self.bottleneck_creator.return_bottleneck(image_path)
            else:
                if self.test_modus or self.mirrored or self.object_proposal_detector == 'faster_rcnn':
                    sub_folder = 0
                else:
                    sub_folder = randint(0, 5)
                bottleneck_folder = os.path.join(self.data_path, self.scene_name, cropped_object_proposals_folder + '_bottlenecks')
                return np.load(os.path.join(bottleneck_folder, str(sub_folder), file_name[:-4] + '.npy'))
        else:
            return np.zeros(2048)

    def load_bottleneck(self):
        return np.load(os.path.join(self.data_path, self.scene_name, 'bottleneck', self.cur_image_name[:-4] + '.npy'))

    def human_render(self):
        obs = self._render()
        title = self.cur_image_name

        gt_boundary_box = None
        best_matching_object_available = False
        for box in self.annotations[self.cur_image_name]['bounding_boxes']:
            if box[4] == self.best_matching_object:
                gt_boundary_box = self.scale_values(box, 4)
                best_matching_object_available = True
            elif box[4] in self.ids and not best_matching_object_available:
                gt_boundary_box = self.scale_values(box, 4)
        return obs, title, gt_boundary_box, self.pred_boundary_box, self.get_target_image()

    def create_observation(self):
        obs = []
        if not self.use_bottlenecks:
            obs.append(self._render())
        else:
            obs.append(self.load_bottleneck())
        get_target_object_vector = self.get_target_object_vector()
        center_x, center_y, size, certainty, _, _, _, _ = self.get_object_candidates()
        if self.remove_classifier_certainty:
            certainty = [0.0] * self.nof_objects
        obs.append(np.column_stack((get_target_object_vector, center_x, center_y, size, certainty)))
        obs.append(self.get_allowed_actions())
        obs.append(self.get_target_object_image_section())

        return obs

    def pose_was_already_visited(self, pose):
        return pose in self.set_of_visited_poses

    def get_allowed_actions(self):
        allowed_actions = [1] * 7

        if self.annotations[self.cur_image_name]['forward'] == '':
            allowed_actions[1] = 0
        if self.annotations[self.cur_image_name]['backward'] == '':
            allowed_actions[2] = 0
        if self.annotations[self.cur_image_name]['left'] == '':
            allowed_actions[3] = 0
        if self.annotations[self.cur_image_name]['right'] == '':
            allowed_actions[4] = 0
        return allowed_actions

    def best_move_towards_object(self, object_name=None):
        if object_name is None:
            object_name = self.target_object_name
        object_id = instance_names[object_name]-1
        _, _, _, certainty, _, _, _, _ = self.get_object_candidates(image=self.cur_image_name)
        max_certainty = certainty[object_id]
        best_move = None
        action = [0, 0, 0]

        if self.annotations[self.cur_image_name]['forward'] != '':
            _, _, _, certainty, _, _, _, _ = self.get_object_candidates(image=self.annotations[self.cur_image_name]['forward'])
            if certainty[object_id] > max_certainty:
                max_certainty = certainty[object_id]
                best_move = 'forward'
                action[0] = 1
        if self.annotations[self.cur_image_name]['backward'] != '':
            _, _, _, certainty, _, _, _, _ = self.get_object_candidates(image=self.annotations[self.cur_image_name]['backward'])
            if certainty[object_id] > max_certainty:
                max_certainty = certainty[object_id]
                best_move = 'backward'
                action[0] = 2
        if self.annotations[self.cur_image_name]['left'] != '':
            _, _, _, certainty, _, _, _, _ = self.get_object_candidates(image=self.annotations[self.cur_image_name]['left'])
            if certainty[object_id] > max_certainty:
                max_certainty = certainty[object_id]
                best_move = 'left'
                action[0] = 3
        if self.annotations[self.cur_image_name]['right'] != '':
            _, _, _, certainty, _, _, _, _ = self.get_object_candidates(image=self.annotations[self.cur_image_name]['right'])
            if certainty[object_id] > max_certainty:
                max_certainty = certainty[object_id]
                best_move = 'right'
                action[0] = 4
        if self.annotations[self.cur_image_name]['rotate_ccw'] != '':
            _, _, _, certainty, _, _, _, _ = self.get_object_candidates(image=self.annotations[self.cur_image_name]['rotate_ccw'])
            if certainty[object_id] > max_certainty:
                max_certainty = certainty[object_id]
                best_move = 'rotate_ccw'
                action[1] = 1
        if self.annotations[self.cur_image_name]['rotate_cw'] != '':
            _, _, _, certainty, _, _, _, _ = self.get_object_candidates(image=self.annotations[self.cur_image_name]['rotate_cw'])
            if certainty[object_id] > max_certainty:
                max_certainty = certainty[object_id]
                best_move = 'rotate_cw'
                action[1] = 2

        return best_move, action

    def get_AVD_baseline_action(self):
        action = [0, 0, 0]
        certainty = 0.0
        _, _, _, certainties, x1, y1, x2, y2 = self.get_object_candidates(scale=False)

        if x1[self.target_object_idx] >= 0:
            certainty = certainties[self.target_object_idx]
            for box in self.annotations[self.cur_image_name]['bounding_boxes']:
                if (x1[self.target_object_idx] != box[0]) or \
                   (y1[self.target_object_idx] != box[1]) or \
                   (x2[self.target_object_idx] != box[2]) or \
                   (y2[self.target_object_idx] != box[3]):
                    continue
                else:
                    if len(box) >= 7:
                        if box[6] == 1:    # move forwards
                            action[0] = 1
                        elif box[6] == 2:  # move backwards
                            action[0] = 2
                        elif box[6] == 3:  # move left
                            action[0] = 3
                        elif box[6] == 4:  # move right
                            action[0] = 4
                        elif box[6] == 5:  # rotate right
                            action[1] = 2
                        elif box[6] == 6:  # rotate left
                            action[1] = 1
        return action, certainty

    def get_target_object_vector(self):
        target_object_vector = [0] * self.nof_objects
        #for idx in self.ids:
        # we are still searching for this particular object, even though other objects might be accepted as proposals
        target_object_vector[self.target_object_id - 1] = 1
        return target_object_vector

    def get_target_scene_vector(self):
        target_scene_vector = [0] * (len(scene_list) * 2)
        for idx in range(len(scene_list)):
            if scene_list[idx] == self.scene_name[0:8]:
                if self.mirrored:
                    target_scene_vector[idx+len(scene_list)] = 1
                else:
                    target_scene_vector[idx] = 1
        return target_scene_vector

    def get_object_candidates(self, scale=True, image=None):
        if image is None:
            image = self.cur_image_name

        center_x = [-1] * self.nof_objects
        center_y = [-1] * self.nof_objects
        size = [-1] * self.nof_objects
        certainty = [0.0] * self.nof_objects
        x1 = [-1] * self.nof_objects
        y1 = [-1] * self.nof_objects
        x2 = [-1] * self.nof_objects
        y2 = [-1] * self.nof_objects

        if self.object_proposal_detector == 'gt':
            key = u'good_detector_proposals'
        elif self.object_proposal_detector == 'faster_rcnn':
            key = u'faster_rcnn_proposals'
        else:
            print("Please select a valid object detector!")

        if key in self.annotations[image]:
            center_x = self.annotations[image][key][0]
            center_y = self.annotations[image][key][1]
            size = self.annotations[image][key][2]
            certainty = self.annotations[image][key][3]
            x1 = self.annotations[image][key][4]
            y1 = self.annotations[image][key][5]
            x2 = self.annotations[image][key][6]
            y2 = self.annotations[image][key][7]
            if scale:
                x1 = self.scale_values(x1, self.nof_objects)
                y1 = self.scale_values(y1, self.nof_objects)
                x2 = self.scale_values(x2, self.nof_objects)
                y2 = self.scale_values(y2, self.nof_objects)

        return center_x, center_y, size, certainty, x1, y1, x2, y2


    def get_map(self):
        image_structs_path = os.path.join(self.data_path, self.scene_name, 'image_structs.mat')
        image_structs = sio.loadmat(image_structs_path)
        image_structs = image_structs['image_structs']
        image_structs = image_structs[0]
        return image_structs

    def get_target_image(self):
        return misc.imread(os.path.join(self.data_path, "BigBIRD_instances", self.target_object_name+".jpg"))

    def define_target_boundary(self, target_boundary_box, internal_call=False):
        if not internal_call:
            self.target_boundary_defined = True
        if target_boundary_box is not None:
            if target_boundary_box[0] >= 0 and target_boundary_box[1] >= 0 and target_boundary_box[2] <= WIDTH and \
                            target_boundary_box[3] <= HEIGHT:
                self.pred_boundary_box = target_boundary_box
            else:
                print("Error: Bounding box not inside image dimensions")
        else:
            self.pred_boundary_box = None

    def calculate_IoU(self, box1, box2):
        box1_area = self.area(box1[2] - box1[0], box1[3] - box1[1])
        # print("box1_area:", box1_area)
        box2_area = self.area(box2[2] - box2[0], box2[3] - box2[1])
        # print("box2_area:", box2_area)
        intersect_area = self.intersection_area(box1[0], box1[3], box1[2], box1[1], box2[0], box2[3], box2[2], box2[1])
        # print("intersect_area:", intersect_area)
        union_area = box1_area + box2_area - intersect_area
        # print("union:", union_area)
        return intersect_area / union_area

    def area(self, width, height):
        return width * height

    # xmin_1 - minimum x value of bounding box 1
    # xmax_1 - maximum x value of bounding box 1
    # ymin_1 - minimum y value of bounding box 1
    # ymax_1 - maximum y value of bounding box 1
    # xmin_2 - minimum x value of bounding box 2
    # xmax_2 - maximum x value of bounding box 2
    # ymin_2 - minimum y value of bounding box 2
    # ymax_2 - maximum y value of bounding box 2
    def intersection_area(self, xmin_1, ymax_1, xmax_1, ymin_1, xmin_2, ymax_2, xmax_2, ymin_2):
        dx = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
        dy = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0

    def scale_values(self, values, length=1):
        scaled_values = []
        for i in range(length):
            scaled_values.append(int(values[i] * self.scale_factor))
        return scaled_values

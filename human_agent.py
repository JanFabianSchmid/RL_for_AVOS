import math
from config import config
from ActiveVisualObjectSearch import ActiveVisualObjectSearchEnv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RectangleSelector
import numpy as np
import sys
# from gym.wrappers.time_limit import TimeLimit

WIDTH = config["image_size"][0]
HEIGHT = config["image_size"][1]
COLORS = config["image_size"][2]

object_detector = "instance_classifier"
for idx in range(len(sys.argv)-1):
    if sys.argv[idx+1] == 'gt':
        object_detector = 'gt'
    elif sys.argv[idx+1] == 'instance':
        object_detector = 'instance_classifier'
    elif sys.argv[idx + 1] == 'faster_rcnn':
        object_detector = 'faster_rcnn'

def pre_process(observation):
    mean = 0.0
    std = 5.0
    noisy = observation + np.random.normal(mean, std, observation.shape)
    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype(np.uint8)
    return noisy

class human_agent:
    def __init__(self):
        self.env = ActiveVisualObjectSearchEnv()
        self.env.set_agent("human")
        self.env.set_target_object_name('aunt_jemima_original_syrup')
        self.env.set_scene_name('Home_001_1')
        self.env.set_first_image('000110014880101.jpg')
        self.env.set_test_modus(True)
        self.env.set_postfix("_human")
        self.env.reset()
        self.action = [0, 0, 0]
        self.proposal_box = None
        self.map_X = []
        self.map_Y = []
        self.directions = []
        self.orientations = []
        self.image_names = []
        self.title = ''

        plt.figure(figsize=(12, 9), dpi=200, facecolor='w', edgecolor='k')
        plt.subplots_adjust(top=0.99, right=0.99, bottom=0.01, left=0.01) # remove white space
        gs = gridspec.GridSpec(4, 4)
        self.ax = plt.subplot(gs[1:, :])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            left='off',
            labelleft='off',
            labelbottom='off')  # labels along the bottom edge are off
        self.ax_map = plt.subplot(gs[0, 1])
        self.ax_map.axis('equal')
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            left='off',
            labelleft='off',
            labelbottom='off')  # labels along the bottom edge are off
        self.ax_target = plt.subplot(gs[0, 2])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            left='off',
            labelleft='off',
            labelbottom='off')  # labels along the bottom edge are off

        #self.fig, (self.ax_target, self.ax) = plt.subplots(1, 2, figsize=(12, 9), dpi=200, facecolor='w', edgecolor='k',
        #                                                   gridspec_kw={'width_ratios': [1, 8]})

        plt.subplot
        self.RS = RectangleSelector(self.ax, self.line_select_callback,
                                    drawtype='box', useblit=True,
                                    button=[1, 3],  # don't use middle button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=False)
        mpl.rcParams['keymap.save'].remove('s')
        mpl.rcParams['keymap.quit'].remove('q')
        plt.connect('key_press_event', self.toggle_selector)
        self.get_map_data()
        self.first = True
        self.render()

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 < 0:
            x1 = 0
        if x2 > WIDTH:
            x2 = WIDTH
        if y1 < 0:
            y1 = 0
        if y2 > HEIGHT:
            y2 = HEIGHT
        self.proposal_box = [x1, y1, x2, y2]
        print("Proposed box:", "(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        self.render()

    def toggle_selector(self, event):
        #print('Key pressed.')
        if event.key in ['C', 'c']:
            print('Close')
            plt.close()
        else:
            if event.key in ['X', 'x']:# and self.RS.active:
                #if self.proposal_box is not None:
                print('Use rectangle as proposal')
                self.action[2] = 1
               # else:
               #     print('Please select an area of the scene as proposal for the target object')
            if event.key in ['H', 'h']:
                print(("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n").format(
                    "Enter a character to move around the scene:",
                    "'w' - move forward",
                    "'a' - move left",
                    "'s' - move backward",
                    "'d' - move right",
                    "'q' - rotate counter clockwise",
                    "'e' - rotate clockwise",
                    "'x' - use current rectangle as proposal"
                    "'c' - quit",
                    "'h' - print this help menu"))
            if event.key in ['W', 'w']:
                self.action[0] = 1
            if event.key in ['S', 's']:
                self.action[0] = 2
            if event.key in ['A', 'a']:
                self.action[0] = 3
            if event.key in ['D', 'd']:
                self.action[0] = 4
            if event.key in ['Q', 'q']:
                self.action[1] = 1
            if event.key in ['E', 'e']:
                self.action[1] = 2

            self.make_step()

    def make_step(self):
        if self.proposal_box is not None:
            self.env.define_target_boundary(self.proposal_box)
        obs, reward, terminal, info = self.env.step(self.action)
        #print("Reward: ", reward)
        #print("AVD step: ", self.env.get_AVD_baseline_action()[0])
        if terminal:
            self.env.reset()
            self.first = True
        self.action = [0, 0, 0]
        self.proposal_box = None
        self.render()

    def get_map_data(self):
        image_structs = self.env.get_map()

        for camera in image_structs:
            # get 3D camera position in the reconstruction
            # coordinate frame
            world_pos = camera[3]

            # get 3D vector that indicates camera viewing direction
            # Add the world_pos to translate the vector from the origin
            # to the camera location.
            direction = world_pos + camera[4]

            self.directions.append(direction)
            self.orientations.append(camera[4])
            self.image_names.append(camera[0][0])
            self.map_X.append(world_pos[0])
            self.map_Y.append(world_pos[2])

    def plot_current_pose(self):
        self.ax_map.cla()
        # plot only 2D, as all camera heights are the same
        self.ax_map.scatter(self.map_X, self.map_Y, marker='o', c='r', s=1)

        for i in range(len(self.image_names)):
            if self.image_names[i] == self.title:
                #print('X:', self.map_X[i], 'Y:', self.map_Y[i])
                #print('Orientation:', str(self.orientations[i]))
                val = 0.0
                for idx in range(3):
                    val += self.orientations[i][idx] * self.orientations[i][idx]
                #print(str(math.sqrt(val)))
                self.ax_map.scatter(self.map_X[i], self.map_Y[i], marker='o', c='b', s=2.5)
                self.ax_map.plot([self.map_X[i], self.directions[i][0]], [self.map_Y[i], self.directions[i][2]], 'b-')

    def plot_target(self, target):
        #target = self.env.get_target_image()
        self.ax_target.imshow(target)

    def render(self):
        #print("Number of steps: ", self.env.nof_steps)
        #print("Steps to target visibility: ", self.env.distanceToPoseWithTargetVisibility())
        #print("Best move: ", self.env.best_move_towards_object(self.env.target_object_name))
        print("Distance to target:", str(self.env.current_distance))
        if self.env.first_target_object_encounter is not None:
            print("object explored!")

        obs, self.title, gt_boundary_box, pred_boundary_box, target = self.env.human_render()
        self.ax.cla()
        obs = pre_process(obs)
        self.ax.imshow(obs)
        print(self.title)
        if gt_boundary_box is not None:
            x_min = gt_boundary_box[0]
            y_min = gt_boundary_box[1]
            x_max = gt_boundary_box[2]
            y_max = gt_boundary_box[3]
            rect = patches.Rectangle(xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
            #self.ax.add_patch(rect)
        if True:
            self.env.set_object_detector(object_detector)
            #self.env.set_use_bottlenecks(False)
            center_x, center_y, size, certainty, x1, y1, x2, y2 = self.env.get_object_candidates()
            target_object_vector = self.env.get_target_object_vector()
            target_object_idx = max(enumerate(target_object_vector), key=lambda t: t[1])[0]

            print("Current target object certainty: " + str(certainty[target_object_idx]))

            #if x1[target_object_idx] >= 0:
            #    proposal_box = [x1[target_object_idx], y1[target_object_idx], x2[target_object_idx], y2[target_object_idx]]
            #    self.env.define_target_boundary(proposal_box)
            #    self.proposal_box = proposal_box
            if size[target_object_idx] > 0:
                #print("Current certainty: ", certainty[target_object_idx])
                x_min = (center_x[target_object_idx] * WIDTH - math.sqrt(size[target_object_idx]*(WIDTH*HEIGHT/4)) / 2)
                y_min = (center_y[target_object_idx] * HEIGHT - math.sqrt(size[target_object_idx]*(WIDTH*HEIGHT/4)) / 2)
                x_max = (center_x[target_object_idx] * WIDTH + math.sqrt(size[target_object_idx]*(WIDTH*HEIGHT/4)) / 2)
                y_max = (center_y[target_object_idx] * HEIGHT + math.sqrt(size[target_object_idx]*(WIDTH*HEIGHT/4)) / 2)

                #x_min = x1[target_object_idx]
                #y_min = y1[target_object_idx]
                #x_max = x2[target_object_idx]
                #y_max = y2[target_object_idx]

                success, _, _ = self.env.check_proposal([x_min, y_min, x_max, y_max])
                print("Would have been successful: ", success)
                rect = patches.Rectangle(xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min,
                                         linewidth=2, edgecolor='g', facecolor='none')
                #self.ax.add_patch(rect)

            for target_object_idx in range(33):
                if size[target_object_idx] > 0:
                    #print("Current certainty: ", certainty[target_object_idx])
                    x_min = (
                    center_x[target_object_idx] * WIDTH - math.sqrt(size[target_object_idx] * (WIDTH * HEIGHT / 4)) / 2)
                    y_min = (center_y[target_object_idx] * HEIGHT - math.sqrt(
                        size[target_object_idx] * (WIDTH * HEIGHT / 4)) / 2)
                    x_max = (
                    center_x[target_object_idx] * WIDTH + math.sqrt(size[target_object_idx] * (WIDTH * HEIGHT / 4)) / 2)
                    y_max = (center_y[target_object_idx] * HEIGHT + math.sqrt(
                        size[target_object_idx] * (WIDTH * HEIGHT / 4)) / 2)

                    # x_min = x1[target_object_idx]
                    # y_min = y1[target_object_idx]
                    # x_max = x2[target_object_idx]
                    # y_max = y2[target_object_idx]

                    success, _, _ = self.env.check_proposal([x_min, y_min, x_max, y_max])
                    #print("Would have been successful: ", success)
                    rect = patches.Rectangle(xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min,
                                             linewidth=2, edgecolor='r', facecolor='none')
                    self.ax.add_patch(rect)

        if self.proposal_box is not None:
            x_min = self.proposal_box[0]
            y_min = self.proposal_box[1]
            x_max = self.proposal_box[2]
            y_max = self.proposal_box[3]
            rect = patches.Rectangle(xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min,
                                     linewidth=2, edgecolor='b', facecolor='none')
            self.ax.add_patch(rect)

        self.plot_current_pose()

        if self.first:
            self.first = False
            self.plot_target(target)
            plt.show()
            plt.pause(0.001)
        else:
            plt.draw()
            plt.pause(0.001)


if __name__ == "__main__":
    agent = human_agent()

import sys
import math
from config import config
from ActiveVisualObjectSearch import ActiveVisualObjectSearchEnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

WIDTH = config["image_size"][0]
HEIGHT = config["image_size"][1]
COLORS = config["image_size"][2]

class human_agent:
    def __init__(self, data_path):
        self.data_path = data_path
        stats = None
        with open(data_path+'/stats', mode='r') as file:
            for line in file:
                stats = line.split(',')

        self.env = ActiveVisualObjectSearchEnv()
        self.env.set_agent("human")
        self.env.set_scene_name(stats[3])
        self.env.set_target_object_name(stats[4])
        self.env.set_first_image(stats[5])
        self.env.set_use_bottlenecks(True)
        self.env.reset()
        self.action = [0, 0, 0]
        self.proposal_box = None
        self.QValue = None
        self.map_X = []
        self.map_Y = []
        self.directions = []
        self.image_names = []
        self.title = ''
        self.frameshistory = []
        self.starting_pose = None
        self.end_pose = None
        self.visited_poses = []
        self.count = 0


        self.fig = plt.figure(figsize=(12, 9), dpi=200, facecolor='w', edgecolor='k')
        self.canvas = FigureCanvas(self.fig)
        plt.subplots_adjust(top=0.99, right=0.99, bottom=0.01, left=0.01)  # remove white space
        gs = gridspec.GridSpec(4, 4)
        self.ax = plt.subplot(gs[1:, :])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            left='off',
            labelleft='off',
            labelbottom='off')  # labels along the bottom edge are off
        self.ax_QValues = plt.subplot(gs[0, 0:2])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left='off',
            labelleft='off')
        self.ax_map = plt.subplot(gs[0, 2])
        self.ax_map.axis('equal')
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            left='off',
            labelleft='off',
            labelbottom='off')  # labels along the bottom edge are off
        self.ax_target = plt.subplot(gs[0, 3])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            left='off',
            labelleft='off',
            labelbottom='off')  # labels along the bottom edge are off

        #self.ax_description = plt.subplot(gs[0, 3])
        #plt.tick_params(
        #    axis='both',  # changes apply to the x-axis
        #    which='both',  # both major and minor ticks are affected
        #    bottom='off',  # ticks along the bottom edge are off
        #    left='off',
        #    labelleft='off',
        #    labelbottom='off')  # labels along the bottom edge are off

        #self.fig, (self.ax_target, self.ax) = plt.subplots(1, 2, figsize=(12, 9), dpi=200, facecolor='w', edgecolor='k',
        #                                                   gridspec_kw={'width_ratios': [1, 8]})

        plt.subplot
        self.get_map_data()
        self.first = True
        self.render()

        first = True
        with open(data_path+'/log', mode='r') as file:
            for line in file:
                if first:
                    first = False
                else:
                    self.action = [0, 0, 0]
                    line = line.strip()
                    step = line.split(';')
                    action = step[0][1:-1].split(',')
                    action[0] = int(action[0])
                    action[1] = int(action[1])
                    action[2] = int(action[2])
                    self.action = action
                    proposal_box = step[3]
                    if proposal_box != 'None':
                        proposal_box = proposal_box[1:-1].split(',')
                        proposal_box[0] = float(proposal_box[0])
                        proposal_box[1] = float(proposal_box[1])
                        proposal_box[2] = float(proposal_box[2])
                        proposal_box[3] = float(proposal_box[3])
                        self.proposal_box = proposal_box
                    else:
                        self.proposal_box = None
                    self.make_step()
                    QValue = step[7]
                    print(QValue)
                    if QValue != 'None' and QValue != '-1.0':
                        QValue = QValue[1:-1].split(',')
                        QValue[0] = float(QValue[0])
                        QValue[1] = float(QValue[1])
                        QValue[2] = float(QValue[2])
                        QValue[3] = float(QValue[3])
                        QValue[4] = float(QValue[4])
                        QValue[5] = float(QValue[5])
                        QValue[6] = float(QValue[6])
                        self.QValue = tuple(QValue)
                        #print("QValue:", self.QValue)

                    self.render()
                    #self.make_step()

                    #if proposal_box != 'None':
                    #    proposal_box = proposal_box[1:-1].split(',')
                    #    proposal_box[0] = float(proposal_box[0])
                    #    proposal_box[1] = float(proposal_box[1])
                    #    proposal_box[2] = float(proposal_box[2])
                    #    proposal_box[3] = float(proposal_box[3])
                    #    self.proposal_box = proposal_box
                    #else:
                    #    self.proposal_box = None

                    self.count += 1
        self.render()
        self.display_frames_as_gif(self.frameshistory, data_path + '/visualization' + '.gif')

    def display_frames_as_gif(self, frames, filename_gif=None):
        """
        Displays a list of frames as a gif, with controls
        """
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        if filename_gif:
            anim.save(filename_gif, writer='imagemagick', fps=0.7)

    def make_step(self):
        self.env.define_target_boundary(self.proposal_box)
        if self.starting_pose is None:
            self.starting_pose = self.title
        obs, reward, terminal, info = self.env.step(self.action)
        print("Reward: ", reward)
        self.end_pose = self.title
        self.visited_poses.append(self.title)

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
            self.image_names.append(camera[0][0])
            self.map_X.append(world_pos[0])
            self.map_Y.append(world_pos[2])

    def plot_current_pose(self):
        self.ax_map.cla()
        # plot only 2D, as all camera heights are the same
        self.ax_map.scatter(self.map_X, self.map_Y, marker='o', c='black', s=6)

        for i in range(len(self.image_names)):
            if self.image_names[i] == self.title:
                self.ax_map.scatter(self.map_X[i], self.map_Y[i], marker='o', c='r', s=20)
                self.ax_map.plot([self.map_X[i], self.directions[i][0]], [self.map_Y[i], self.directions[i][2]], 'r-',linewidth=3)

    def render(self):
        print(self.env.cur_image_name)
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 11,
                }

        self.title = self.env.cur_image_name
        obs = self.env._render()
        self.ax.cla()
        self.ax.imshow(obs)
        #self.ax.title("Action")

        if True:
            #self.env.set_object_detector('gt')
            center_x, center_y, size, certainty, x1, y1, x2, y2 = self.env.get_object_candidates()
            scale_factor = config["original_image_size"][0] / config["image_size"][0]
            target_object_vector = self.env.get_target_object_vector()
            target_object_idx = max(enumerate(target_object_vector), key=lambda t: t[1])[0]
            #if x1[target_object_idx] >= 0:
            #    proposal_box = [x1[target_object_idx], y1[target_object_idx], x2[target_object_idx], y2[target_object_idx]]
            #    self.env.define_target_boundary(proposal_box)
            #    self.proposal_box = proposal_box
            if size[target_object_idx] > 0:
                #print(certainty[target_object_idx])
                x_min = center_x[target_object_idx] - math.sqrt(size[target_object_idx])/2
                y_min = center_y[target_object_idx] - math.sqrt(size[target_object_idx])/2
                x_max = center_x[target_object_idx] + math.sqrt(size[target_object_idx])/2
                y_max = center_y[target_object_idx] + math.sqrt(size[target_object_idx])/2
                rect = patches.Rectangle(xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min,
                                         linewidth=2, edgecolor='g', facecolor='none')
                #self.ax.add_patch(rect)

        if self.proposal_box is not None:
            x_min = self.proposal_box[0] * scale_factor
            y_min = self.proposal_box[1] * scale_factor
            x_max = self.proposal_box[2] * scale_factor
            y_max = self.proposal_box[3] * scale_factor
            rect = patches.Rectangle(xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min,
                                     linewidth=2, edgecolor='b', facecolor='none')
            self.ax.add_patch(rect)

        #actions = ('Prop', 'F', 'B', 'L', 'R', 'CCW', 'CW')
        actions = ('Found', 'Forw.', 'Backw.', 'Left', 'Right', 'CCW', 'CW')
        index = np.arange(len(actions))
        self.ax_QValues.cla()
        QValue = self.QValue
        if QValue is None:
            QValue = (0, 0, 0, 0, 0, 0, 0)
        self.ax_QValues.bar(index, QValue, align='center', alpha=0.5)
        self.ax_QValues.set_ylim([0, 1])
        self.ax_QValues.set_xticks(index)
        self.ax_QValues.set_xticklabels(actions, fontdict=font)

        action_text = ''
        if self.action[0] == 1:
            action_text = '     Forward'
        elif self.action[0] == 2:
            action_text = '     Backward'
        elif self.action[0] == 3:
            action_text = '    Move Left'
        elif self.action[0] == 4:
            action_text = '    Move Right'
        elif self.action[1] == 1:
            action_text = '    Rotate CCW'
        elif self.action[1] == 2:
            action_text = '    Rotate CW'
        elif self.action[2] == 1:
            action_text = ' Make Target Proposal'

        #self.ax_description.cla()
        #self.ax_description.text(0, 4, action_text, fontsize=16, fontdict=font)
        #self.ax_description.axis([0, 10, 0, 10])

        self.plot_current_pose()

        self.fig.canvas.draw()
        buf = self.fig.canvas.tostring_rgb()
        ncols, nrows = self.fig.canvas.get_width_height()
        img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

        self.frameshistory.append(img)
        plt.savefig(self.data_path + '/vis_' + str(self.count) + '.png')

if __name__ == "__main__":
    agent = human_agent(sys.argv[1])

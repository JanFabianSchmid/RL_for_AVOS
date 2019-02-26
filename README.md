# RL_for_AVOS
Code for Active Visual Object Search (AVOS) Using Deep Reinforcement Learning

## Requirements
- Tensorflow 1.4.0 for Python3
- [Download](https://drive.google.com/open?id=1VflwmxfLVTee7fDm_eR2xGz9CrrerdzF) and extract the data into the repository folder, it contains the pre-computed outputs of the instance classifier of Ammirato et al., the required outputs of the Inception v3 network, as well as the object proposals from Faster RCNN.

## Use of our pre-trained networks
To use our DRQN agent trained with Faster R-CNN (RPN) object proposals:
- [Download](https://drive.google.com/open?id=1EfgheF2GyoyHvwWrngsKoNRxhmZtfyDw) savedweights_pretrained_faster_rcnn_drqn into this folder
- execute:  $python3 RL_agent.py _pretrained_faster_rcnn_drqn faster_rcnn drqn

To use our DRQN agent trained with ground truth (GT) object proposals:
- [Download](https://drive.google.com/open?id=1mrWDXDls5d_Z6SjA7OitOlh6q2yfEz5r) savedweights_pretrained_gt_drqn into this folder
- execute:  $python3 RL_agent.py _pretrained_gt_drqn gt drqn

## Training new agents
- To train DRQN using Faster R-CNN object proposals: $python3 RL_agent.py _faster_rcnn_drqn faster_rcnn drqn
- To train DRQN using GT object proposals: $python3 RL_agent.py _gt_drqn gt drqn

### The following scans are used during training
Home_001_1, Home_002_1, Home_003_1, Home_004_1, Home_004_2, Home_005_1, Home_006_1, Home_007_1,
Home_008_1, Home_011_1, Home_013_1, Home_014_1, Home_014_2, Home_016_1

### The following target objects are used during training
advil_liqui_gels, aunt_jemima_original_syrup, cholula_chipotle_hot_sauce,
nature_valley_sweet_and_salty_nut_roasted_mix_nut, coca_cola_glass_bottle,
hersheys_bar, nature_valley_sweet_and_salty_nut_peanut, listerine_green,
mahatma_rice, nutrigrain_harvest_blueberry_bliss, softsoap_clear, softsoap_gold,
softsoap_white, spongebob_squarepants_fruit_snaks, nature_valley_sweet_and_salty_nut_cashew,
tapatio_hot_sauce, vo5_tea_therapy_healthful_green_tea_smoothing_shampoo, 
nature_valley_granola_thins_dark_chocolate, red_cup, expo_marker_red, hunts_sauce,
honey_bunches_of_oats_honey_roasted, honey_bunches_of_oats_with_almonds, pepto_bismol,
pringles_bbq, progresso_new_england_clam_chowder, red_bull

## Evaluation of the agents
We split test scenes into partially known scenes and unknown scenes.
- Partially known scenes have been used with a different object arragentment during training (easy - Home_005_2, medium - Home_001_2, hard - Home_003_2).
- Unknown scenes have never been visited during training (easy - Home_015_01, hard - Home_010_01).

Furthermore, we split test target objects into familiar objects and unfamiliar objects:
- Familiar objects have been searched for during training (they are at different positions during testing for partially known scenes). During testing we use the following familiar objects as target objects: aunt_jemima_original_syrup, mahatma_rice, coca_cola_glass_bottle, spongebob_squarepants_fruit_snaks, tapatio_hot_sauce
- Unfamiliar objects have not been used as search targets during training. During testing we use the following unfamiliar objects as target objects: crystal_hot_sauce, bumblebee_albacore, quaker_chewy_low_fat_chocolate_chunk, nature_valley_sweet_and_salty_nut_almond, crest_complete_minty_fresh

The default scenario of the examined AVOS task is the search for familiar objects in partially known scenes, which represents the task that a robot is faced when searching for targets in a known environment.

In order to test on familiar scenes with familiar objects
- Use the same parameters as for training and add the parameter test_on_familiar_scenes_with_familiar_objects to it
-- E.g. to test drqn trained on GT proposals (with weights stored in savedweights_gt_drqn): $python3 RL_agent.py _gt_drqn gt drqn test_on_familiar_scenes_with_familiar_objects
- To test on unfamiliar scenes with familiar objects add the parameter: test_on_unfamiliar_scenes_with_familiar_objects
- To test on familiar scenes with unfamiliar objects add the parameter: test_on_familiar_scenes_with_unfamiliar_objects
- Results will be written to test_series_results_x_nameofyouragent

## Visualization of the agent behavior
Download the active vision dataset: http://cs.unc.edu/~ammirato/active_vision_dataset_website/get_data.html
- Merge the extracted content (the folder ActiveVisionDataset) with the ActiveVisionDataset folder you downloaded into this repository previously

Run the agent with the parameter "log"
- e.g. $python3 RL_agent.py _gt_drqn gt drqn log
- the log is written to logs_nameofyouragent (e.g. logs_gt_drqn)
- this folder contains one log folder for each search episode performed by the agent
- execute: $python3 visualize_log.py path/to/log/folder

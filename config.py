config = {
    "statistics_file_path": "stats",              # file path for the statistics
    "log_folder_path": "logs",                    # file path for the log folder
    "perfect_approaching": False,
    "perfect_approaching_proposal_test": False,                              # should each step be logged?
    "max_steps": 125,                              # maximum number of steps in one episode
    "success_reward": 50,                          # reward for finding the target object
    "entering_proposal_position_reward": 10,
    "leaving_proposal_position_punishment": -10,
    "staying_in_proposal_position_reward": 0,
    "step_punishment": -2,                        # reward for taking another step without proposing a solution
    "wrong_proposal_punishment": -50,             # reward for a wrong target object proposal, should be negative success_reward, otherwise there is bias towards (not) using the proposal
    "illegal_movement_punishment": -5,            # reward for trying to move outside the camera pose grid
    "original_image_size": (1920, 1080, 3),       # image dimensions of the original image
    "image_size": (600, 338, 3),                  # image dimensions
    "render_mode": 'full_resolution',             # render mode
    "allow_not_available_target_objects": False,  # when randomly selecting a target object
    "use_mirrored_training_scenes": False,
    "chance_of_using_mirrored_training_scene": 0.3,
     "train_scene_list": ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_004_2', 'Home_005_1', 'Home_006_1',
     'Home_007_1', 'Home_008_1', 'Home_011_1', 'Home_013_1', 'Home_014_1', 'Home_014_2', 'Home_016_1'], # list of scenes available for training
    "test_scene_list_familiar_scenes": ['Home_001_2', 'Home_003_2', 'Home_005_2'],
    "test_scene_list_unfamiliar_scenes": ['Home_010_1', 'Home_015_1'],
    "use_all_available_objects_in_training": False,      # instead of train_object_list use all available objects
    "use_all_available_objects_in_tests": False,         # instead of test_object_list use all available objects
    "train_object_list": ['advil_liqui_gels', 'aunt_jemima_original_syrup', 'cholula_chipotle_hot_sauce','nature_valley_sweet_and_salty_nut_roasted_mix_nut',
                          'coca_cola_glass_bottle', 'hersheys_bar','nature_valley_sweet_and_salty_nut_peanut',
                          'listerine_green','mahatma_rice', 'nutrigrain_harvest_blueberry_bliss','softsoap_clear','softsoap_gold','softsoap_white',
                          'spongebob_squarepants_fruit_snaks','nature_valley_sweet_and_salty_nut_cashew',
                          'tapatio_hot_sauce', 'vo5_tea_therapy_healthful_green_tea_smoothing_shampoo','nature_valley_granola_thins_dark_chocolate',
                          'red_cup', 'expo_marker_red', 'hunts_sauce', 'honey_bunches_of_oats_honey_roasted','honey_bunches_of_oats_with_almonds',
                          'pepto_bismol','pringles_bbq','progresso_new_england_clam_chowder','red_bull'],  # all but paper plateq
    "test_object_list_familiar_objects": ['aunt_jemima_original_syrup', 'mahatma_rice', 'coca_cola_glass_bottle',
                         'spongebob_squarepants_fruit_snaks', 'tapatio_hot_sauce'],
    "test_object_list_unfamiliar_objects": ['crystal_hot_sauce', 'bumblebee_albacore','quaker_chewy_low_fat_chocolate_chunk',
                         'nature_valley_sweet_and_salty_nut_almond', 'crest_complete_minty_fresh'] # not used during training
}

seen_object_ids = [1,2,4,5,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,29,30,31,33]
unseen_object_ids = [3, 6, 7, 20, 28]

scene_list = ['Home_001', 'Home_002', 'Home_003', 'Home_004', 'Home_005', 'Home_006', 'Home_007', 'Home_008',
              'Home_010', 'Home_011', 'Home_013', 'Home_014',  'Home_015', 'Home_016']

nature_valley = [15, 28, 29, 30, 31]
soap = [22, 23, 24]
honey_bunches = [10, 11]

instance_names = {
    "background": 0,
    "advil_liqui_gels": 1,
    "aunt_jemima_original_syrup": 2,
    "bumblebee_albacore": 3,
    "cholula_chipotle_hot_sauce": 4,
    "coca_cola_glass_bottle": 5,
    "crest_complete_minty_fresh": 6,
    "crystal_hot_sauce": 7,
    "expo_marker_red": 8,
    "hersheys_bar": 9,
    "honey_bunches_of_oats_honey_roasted": 10,
    "honey_bunches_of_oats_with_almonds": 11,
    "hunts_sauce": 12,
    "listerine_green": 13,
    "mahatma_rice": 14,
    "nature_valley_granola_thins_dark_chocolate": 15,
    "nutrigrain_harvest_blueberry_bliss": 16,
    "pepto_bismol": 17,
    "pringles_bbq": 18,
    "progresso_new_england_clam_chowder": 19,
    "quaker_chewy_low_fat_chocolate_chunk": 20,
    "red_bull": 21,
    "softsoap_clear": 22,
    "softsoap_gold": 23,
    "softsoap_white": 24,
    "spongebob_squarepants_fruit_snaks": 25,
    "tapatio_hot_sauce": 26,
    "vo5_tea_therapy_healthful_green_tea_smoothing_shampoo": 27,
    "nature_valley_sweet_and_salty_nut_almond": 28,
    "nature_valley_sweet_and_salty_nut_cashew": 29,
    "nature_valley_sweet_and_salty_nut_peanut": 30,
    "nature_valley_sweet_and_salty_nut_roasted_mix_nut": 31,
    "paper_plate": 32,
    "red_cup": 33
}

# Home_001 x2, good loop-closure
# Home_002   , rather difficult, 3 rooms with small corridors, jumps in between poses
# Home_003 x2, bit difficult, distant bathroom and a large eat-in kitchen
# Home_004 x2, rather difficult, 2 large rooms and a bathroom, very cluttered
# Home_005 x2, easy only a kitchen
# Home_006   , ok loop-closure
# Home_007   , good two connected rooms, loop-closure
# Home_008   , rather easy one large room
# Home_010   , difficult large eat-in kitchen, difficult to navigate, loop-closure
# Home_011   , ok kitchen and living room, loop-closure
# Home_013   , bit difficult large eat-in kitchen and far away second room
# Home_014 x2, good small student dorm with bathroom
# Home_015   , easy only a bedroom
# Home_016   , good kitchen and bathroom

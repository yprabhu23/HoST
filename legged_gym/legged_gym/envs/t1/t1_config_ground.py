from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class T1Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        rot = [0.0, -1, 0, 1.0] # x,y,z,w [quat]
        target_joint_anlges = {}
        default_joint_angles = []
        target_joint_angles = { # = target angles [rad] when action = 0.0
        #    'left_hip_yaw_joint' : 0. ,   
        #    'left_hip_roll_joint' : 0,               
        #    'left_hip_pitch_joint' : -0.1,         
        #    'left_knee_joint' : 0.3,       
        #    'left_ankle_joint' : -0.2,     
        #    'right_hip_yaw_joint' : 0., 
        #    'right_hip_roll_joint' : 0, 
        #    'right_hip_pitch_joint' : -0.1,                                       
        #    'right_knee_joint' : 0.3,                                             
        #    'right_ankle_joint' : -0.2,                                     
        #    'torso_joint' : 0., 
        #    'left_shoulder_pitch_joint' : 0., 
        #    'left_shoulder_roll_joint' : 0.3, 
        #    'left_shoulder_yaw_joint' : 0.,
        #    'left_elbow_joint'  : 0.7,
        #    'right_shoulder_pitch_joint' : 0.,
        #    'right_shoulder_roll_joint' : -0.3,
        #    'right_shoulder_yaw_joint' : 0.,
        #    'right_elbow_joint' : 0.7,

        # Head/neck
        "AAHead_yaw": 0.0,
        "Head_pitch": 0.0,

        # Torso
        "Waist": 0.0,

        # Left arm
        "Left_Shoulder_Pitch": 0.0,
        "Left_Shoulder_Roll": 0.3,
        "Left_Elbow_Pitch": 0.7,
        "Left_Elbow_Yaw": 0.0,
        "Left_Wrist_Pitch": 0.0,
        "Left_Wrist_Yaw": 0.0,
        "Left_Hand_Roll": 0.0,

        # Right arm
        "Right_Shoulder_Pitch": 0.0,
        "Right_Shoulder_Roll": -0.3,
        "Right_Elbow_Pitch": 0.7,
        "Right_Elbow_Yaw": 0.0,
        "Right_Wrist_Pitch": 0.0,
        "Right_Wrist_Yaw": 0.0,
        "Right_Hand_Roll": 0.0,

        # Left leg
        "Left_Hip_Yaw": 0.0,
        "Left_Hip_Roll": 0.0,
        "Left_Hip_Pitch": -0.1,
        "Left_Knee_Pitch": 0.3,
        "Left_Ankle_Pitch": -0.2,
        "Left_Ankle_Roll": 0.0,

        # Right leg
        "Right_Hip_Yaw": 0.0,
        "Right_Hip_Roll": 0.0,
        "Right_Hip_Pitch": -0.1,
        "Right_Knee_Pitch": 0.3,
        "Right_Ankle_Pitch": -0.2,
        "Right_Ankle_Roll": 0.0,

        }


        default_joint_angles = { # = target angles [rad] when action = 0.0
        #    'left_hip_yaw_joint' : 0. ,   
        #    'left_hip_roll_joint' : 0,               
        #    'left_hip_pitch_joint' : -0.1,         
        #    'left_knee_joint' : 0.3,       
        #    'left_ankle_joint' : -0.2,     
        #    'right_hip_yaw_joint' : 0., 
        #    'right_hip_roll_joint' : 0, 
        #    'right_hip_pitch_joint' : -0.1,                                       
        #    'right_knee_joint' : 0.3,                                             
        #    'right_ankle_joint' : -0.2,                                     
        #    'torso_joint' : 0., 
        #    'left_shoulder_pitch_joint' : 0., 
        #    'left_shoulder_roll_joint' : 0, 
        #    'left_shoulder_yaw_joint' : 0.,
        #    'left_elbow_joint'  : 0.,
        #    'right_shoulder_pitch_joint' : 0.,
        #    'right_shoulder_roll_joint' : 0.0,
        #    'right_shoulder_yaw_joint' : 0.,
        #    'right_elbow_joint' : 0.,

        # Head/neck
        "AAHead_yaw": 0.0,
        "Head_pitch": 0.0,

        # Torso
        "Waist": 0.0,

        # Left arm
        "Left_Shoulder_Pitch": 0.0,
        "Left_Shoulder_Roll": 0.3,
        "Left_Elbow_Pitch": 0.7,
        "Left_Elbow_Yaw": 0.0,
        "Left_Wrist_Pitch": 0.0,
        "Left_Wrist_Yaw": 0.0,
        "Left_Hand_Roll": 0.0,

        # Right arm
        "Right_Shoulder_Pitch": 0.0,
        "Right_Shoulder_Roll": -0.3,
        "Right_Elbow_Pitch": 0.7,
        "Right_Elbow_Yaw": 0.0,
        "Right_Wrist_Pitch": 0.0,
        "Right_Wrist_Yaw": 0.0,
        "Right_Hand_Roll": 0.0,

        # Left leg
        "Left_Hip_Yaw": 0.0,
        "Left_Hip_Roll": 0.0,
        "Left_Hip_Pitch": -0.1,
        "Left_Knee_Pitch": 0.3,
        "Left_Ankle_Pitch": -0.2,
        "Left_Ankle_Roll": 0.0,

        # Right leg
        "Right_Hip_Yaw": 0.0,
        "Right_Hip_Roll": 0.0,
        "Right_Hip_Pitch": -0.1,
        "Right_Knee_Pitch": 0.3,
        "Right_Ankle_Pitch": -0.2,
        "Right_Ankle_Roll": 0.0,
        }

    class env(LeggedRobotCfg.env):
        num_one_step_observations=  64 #+ 3 * 2#+ 3 * 11  # +3*11 actions / -3 commands  i
        num_actions = 29#19 #+ 2# + 11
        num_dofs = 29#19
        num_actor_history = 6
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 10 # episode length in seconds
        unactuated_timesteps = 30

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip': 350,
                     'knee': 350,
                     'ankle': 120,
                     'shoulder': 350,
                     'elbow': 350,
                     'torso': 200,
                     }  # [N*m/rad]
        damping = {  'hip': 4,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder': 4,
                     'elbow': 4,
                     'torso': 4,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.7
        restitution = 0.3
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [1, 0., 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/t1/urdf/t1_29dof.urdf'
        name = "t1"
        left_foot_name = "Left_Ankle"
        right_foot_name = "Right_ankle"
        left_knee_name = 'Left_Knee'
        right_knee_name = 'Right_Knee'
        left_thigh_name = 'Left_Hip_Pitch'
        right_thigh_name = 'Right_Hip_Pitch'
        foot_name = "Ankle"
        penalize_contacts_on = ["Elbow", 'Shoulder', 'Trunk', 'Waist', 'Knee', 'Hip']
        terminate_after_contacts_on = []    #'torse'
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

        left_shoulder_name = "left_shoulder"
        right_shoulder_name = "right_shoulder"
        # LEft Hip Yaw 
        left_leg_joints = ['Left_Hip_Yaw', 'Left_Hip_Roll', 'Left_Hip_Pitch', 'Left_Knee_Pitch', 'Left_Ankle_Roll', 'Left_Ankle_Pitch']
        right_leg_joints = ['Right_Hip_Yaw', 'Right_Hip_Roll', 'Right_Hip_Pitch', 'Right_Knee_Pitch',  'Right_Ankle_Roll', 'Right_Ankle_Pitch']
        left_hip_joints = ['Left_Hip_Yaw', 'Left_Hip_Roll', 'Left_Hip_Pitch']
        right_hip_joints = ['Right_Hip_Yaw',  'Right_Hip_Roll', 'Right_Hip_Pitch']

        left_hip_roll_joints = ['Left_Hip_Roll',]
        right_hip_roll_joints = ['Right_Hip_Roll',]    

        left_hip_pitch_joints = ['Left_Hip_Pitch']
        right_hip_pitch_joints = ['Right_Hip_Pitch']     

        left_shoulder_roll_joints = ["Left_Shoulder_Roll"]
        right_shoulder_roll_joints = ["Right_Shoulder_Roll"]    


        left_knee_joints = ['Left_Knee_Pitch']
        right_knee_joints = ['Right_Knee_Pitch']    
        # NEed to edit here
        left_arm_joints = ["Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw","Left_Wrist_Pitch", "Left_Wrist_Yaw" ]
        right_arm_joints = ["Right_Shoulder_Pitch", "Right_Shoulder_Roll", 'Right_Elbow_Pitch', "Right_Elbow_Yaw", "Right_Wrist_Pitch", "Right_Wrist_Yaw", ]
        waist_joints = ["Waist"]
        knee_joints = ['Left_Knee_Pitch', 'Right_Knee_Pitch']
        ankle_joints = ['Left_Ankle_Roll', 'Left_Ankle_Pitch','Right_Ankle_Roll', 'Right_Ankle_Pitch']

        keyframe_name = "keyframe"
        head_name = 'keyframe_head'
        armature = 0

        trunk_names = ['Trunk', 'Waist']
        base_name = 'Trunk'
        # tracking_body_names =  ['pelvis']

        left_upper_body_names = ["Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw","AL1", "AL2", "AL3", "AL4"]
        right_upper_body_names = ["Right_Shoulder_Pitch", "Right_Shoulder_Roll", 'Right_Elbow_Pitch', "Right_Elbow_Yaw","AR1", "AR2", "AR3", "AR4"]
        left_lower_body_names = ['Left_Hip', 'Left_Ankle', 'Left_Knee', "Hip_Pitch_Left", "Hip_Roll_Left", "Hip_Yaw_Left", "Shank_Left"]
        right_lower_body_names = ['Right_Hip', 'Right_Ankle', 'Right_Knee', "Hip_Pitch_Right", "Hip_Roll_Right", "Hip_Yaw_Right", "Shank_Right"]

        left_ankle_names = ['Left_Ankle']
        right_ankle_names = ['Right_Ankle']

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 1
        base_height_sigma = 0.25
        tracking_dof_sigma = 0.25
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        orientation_sigma = 1
        is_gaussian = True
        target_head_height = 1.5
        target_head_margin = 1
        target_base_height_phase1 = 0.65
        target_base_height_phase2 = 0.65
        target_base_height_phase3 = 0.9
        orientation_threshold = 0.99
        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        target_dof_pos_sigma = -0.1
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [2.5, 0.1, 1, 1]

        class scales:
            task_orientation = 1
            task_head_height = 1

    class constraints( LeggedRobotCfg.rewards ):
        is_gaussian = True
        target_head_height = 1.45
        target_head_margin = 1
        orientation_height_threshold = 0.9

        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        hip_yaw_var_sigma = -2
        target_dof_pos_sigma = -0.1
        post_task = False
        
        class scales:
            # regularization reward
            regu_dof_acc = -2.5e-7
            regu_action_rate = -0.01
            regu_smoothness = -0.01 
            regu_torques = -2.5e-6
            regu_joint_power = -2.5e-5
            regu_dof_vel = -1e-3
            regu_joint_tracking_error = -0.00025
            regu_dof_pos_limits = -100.0
            regu_dof_vel_limits = -1 #0.0

            # style reward
            style_waist_deviation = -10
            style_hip_yaw_deviation = -10
            style_hip_roll_deviation = -10
            style_shoulder_roll_deviation = -2.5
            style_left_foot_displacement = 2.5
            style_right_foot_displacement = 2.5
            style_knee_deviation = -0.25
            style_shank_orientation = 10
            style_ground_parallel = 20
            style_feet_distance = -10
            style_style_ang_vel_xy = 1

            # post-task reward
            target_ang_vel_xy = 10
            target_lin_vel_xy = 10
            target_feet_height_var = 2.5
            target_target_upper_dof_pos = 10
            target_target_orientation = 10
            target_target_base_height = 10

    class domain_rand:
        use_random = True

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        randomize_payload_mass = use_random
        payload_mass_range = [-2, 5]

        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 1]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 0.5

        delay = use_random
        max_delay_timesteps = 5

    class curriculum:
        pull_force = True
        force = 400
        threshold_height = 1.4
        dof_vel_limit = 300
        base_vel_limit = 20
        no_orientation = True

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class T1CfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 't1_ground'
        algorithm_class_name = 'PPO'
        init_at_random_ep_len = True
        max_iterations = 12000 # number of policy updates
import header
import importlib
import sys


importlib.reload(header)  # For reloading after making changes
from header import *

import manipulate

# ACTION_TABLE = {
#     "Total Length of Fwd Packet": "Total Length of Fwd Packet",
#     "Fwd Packet Length Mean": "Fwd Packet Length Mean",
#     "Bwd Packets/s": "Bwd Packets/s",
#     "Subflow Fwd Bytes": "Subflow Fwd Bytes",
# }

# ACTION_TABLE = {
#     "TotalLengthofFwdPacket": "TotalLengthofFwdPacket",
#     "FwdPacketLengthMean": "FwdPacketLengthMean",
#     "BwdPackets/s": "BwdPackets/s",
#     "SubflowFwdBytes": "SubflowFwdBytes",
#     "TotalLengthofFwdPacket": "TotalLengthofFwdPacket",
#     "FwdPacketLengthMean": "FwdPacketLengthMean",
#     "BwdPackets/s": "BwdPackets/s",
#     "SubflowFwdBytes": "SubflowFwdBytes",
#     "TotalLengthofFwdPacket": "TotalLengthofFwdPacket",
#     "FwdPacketLengthMean": "FwdPacketLengthMean",
#     "BwdPackets/s": "BwdPackets/s",
#     "SubflowFwdBytes": "SubflowFwdBytes",
# }

# ================================================================
# ISCX-2014 Features
ACTION_TABLE_ISCX_2014 = {
    "FlowDuration": "FlowDuration",
    "FlowBytes/s": "FlowBytes/s",
    "FlowPackets/s": "FlowPackets/s",
    "FwdPackets/s": "FwdPackets/s",
    "BwdPackets/s": "BwdPackets/s",
    "TotalLengthofFwdPacket": "TotalLengthofFwdPacket",
    "TotalLengthofBwdPacket": "TotalLengthofBwdPacket",
    "BwdPackets/s": "BwdPackets/s",
    "SubflowFwdBytes": "SubflowFwdBytes",
    "FwdHeaderLength": "FwdHeaderLength",
    "BwdHeaderLength": "BwdHeaderLength",
    "Down/UpRatio": "Down/UpRatio",
    "AveragePacketSize": "AveragePacketSize",
}

# ISCX-2017 Features
ACTION_TABLE_CIC_2017 = {
    "FlowDuration": "FlowDuration",
    "FlowBytes/s": "FlowBytes/s",
    "FlowPackets/s": "FlowPackets/s",
    "FwdPackets/s": "FwdPackets/s",
    "BwdPackets/s": "BwdPackets/s",
    "TotalLengthofFwdPackets": "TotalLengthofFwdPackets",
    "TotalLengthofBwdPackets": "TotalLengthofBwdPackets",
    "BwdPackets/s": "BwdPackets/s",
    "SubflowFwdBytes": "SubflowFwdBytes",
    "FwdHeaderLength": "FwdHeaderLength",
    "BwdHeaderLength": "BwdHeaderLength",
    "Down/UpRatio": "Down/UpRatio",
    "AveragePacketSize": "AveragePacketSize",
}

# ISCX-2018 Features
ACTION_TABLE_CIC_2018 = {
    "Flow Duration": "Flow Duration",
    "Flow Byts/s": "Flow Byts/s",
    "Flow Pkts/s": "Flow Pkts/s",
    "Fwd Pkts/s": "Fwd Pkts/s",
    "Bwd Pkts/s": "Bwd Pkts/s",
    "TotLen Fwd Pkts": "TotLen Fwd Pkts",
    "TotLen Bwd Pkts": "TotLen Bwd Pkts",
    "Bwd Pkts/s": "Bwd Pkts/s",
    "Subflow Fwd Byts": "Subflow Fwd Byts",
    "Fwd Header Len": "Fwd Header Len",
    "Bwd Header Len": "Bwd Header Len",
    "Down/Up Ratio": "Down/Up Ratio",
    "Pkt Size Avg": "Pkt Size Avg",
}


ACTION_LOOKUP = {i: act for i, act in enumerate(ACTION_TABLE_ISCX_2014.keys())}


class RELEVAGAN_CC(gym.Env):
    def __init__(self, IMG_SHAPE=60):

        # Input shape
        self.img_shape = IMG_SHAPE
        self.num_classes = 1
        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)
        losses = ["binary_crossentropy", "binary_crossentropy", "binary_crossentropy"]
        comb_losses = ["binary_crossentropy", "binary_crossentropy"]
        # tf.compat.v1.disable_eager_execution()

        # Build the generator
        self.generator = self.build_generator()
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses, optimizer=optimizer)
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, bot, normal = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, bot])
        self.combined.compile(loss=comb_losses, optimizer=optimizer)

        # RL Model
        self.action_space = Discrete(len(ACTION_LOOKUP))

        observation_high = np.finfo(np.float32).max

        # self.observation_space = spaces.Box(
        #     low=-observation_high,
        #     high=observation_high,
        #     shape=(IMG_SHAPE,),
        #     dtype=np.float32,
        # ) # 60 for ISCX and 67 for CIC-2017

        self.observation_space = np.zeros(IMG_SHAPE)  # 60 for ISCX and 67 for CIC-2017

        # self.maxturns = maxturns
        self.random_sample = 0
        self.sample_iteration_index = 0

        self.samples = {}

        self.turns = 0

        self.reward = 0

        self.evasion_set = pd.DataFrame({"A": []})

        self.is_first_time = True

        self.agent = 0

        self.previous_weights = 0

        self.weights = 0
        self.d_weights_updated = 0
        self.weights2 = 0

        self.rizwan = 100

        self.evasion_count = 0

        self.minority_training_count = 0

        self.ROUNDS_FACTOR = 1

        self.RL_TRAIN_BATCH = 0

        self.sample_number = -1

        self.nb_actions = 0

        self.cache_path = ""

        self.DATA_SET = ""

        self.agent_nn = 0

        self.temp_sample = 0

        self.XGB = 0

        # ---------------------------------------------------
        # ---------------------------------------------------
        # ---------------------------------------------------

    #     # TODO: Manipulate the input batch as per the random change in the particular action_index

    # def step(self, action_index):

    #     self.observation_space = self.RL_TRAIN_BATCH.iloc[
    #         self.sample_number % self.RL_TRAIN_BATCH.shape[0]
    #     ]

    #     episode_over = False

    #     if self.DATA_SET == "ISCX-2014":
    #         ACTION_LOOKUP = {
    #             i: act for i, act in enumerate(ACTION_TABLE_ISCX_2014.keys())
    #         }
    #     elif self.DATA_SET == "CIC-2017":
    #         ACTION_LOOKUP = {
    #             i: act for i, act in enumerate(ACTION_TABLE_CIC_2017.keys())
    #         }
    #     elif self.DATA_SET == "CIC-2018":
    #         ACTION_LOOKUP = {
    #             i: act for i, act in enumerate(ACTION_TABLE_CIC_2018.keys())
    #         }

    #     assert action_index < len(ACTION_LOOKUP)
    #     action = ACTION_LOOKUP[action_index]

    #     delta = self.RL_TRAIN_BATCH[action[0 : len(action)]].min()

    #     perturbed = (
    #         self.RL_TRAIN_BATCH.iloc[self.sample_number % self.RL_TRAIN_BATCH.shape[0]][
    #             action[0 : len(action)]
    #         ]
    #         + delta
    #     )
    #     self.temp_sample = self.RL_TRAIN_BATCH.copy()

    #     self.temp_sample.iloc[self.sample_number % self.temp_sample.shape[0]][
    #         action[0 : len(action) - 1]
    #     ] = perturbed  # %256 remiander, Assign new value to the selected feature

    #     if(USE_RL_AT_THE_END == False):
    #         pred = self.discriminator.predict(
    #         np.array(
    #             [
    #                 self.temp_sample.iloc[
    #                     self.sample_number % self.temp_sample.shape[0]
    #                 ],
    #             ]
    #         )
    #         )
    #         prediction = pred[1]
    #     else:
    #         pred = predict_clf(
    #             np.array(
    #                 [
    #                     self.temp_sample.iloc[
    #                         self.sample_number % self.temp_sample.shape[0]
    #                     ],
    #                 ]
    #             ),
    #             np.array(
    #                 [
    #                     self.temp_sample.iloc[
    #                         self.sample_number % self.temp_sample.shape[0]
    #                     ],
    #                 ]
    #             ),
    #             np.array(
    #                 [
    #                     self.temp_sample.iloc[
    #                         self.sample_number % self.temp_sample.shape[0]
    #                     ],
    #                 ]
    #             ),
    #             self.XGB,
    #             ONLY_GZ=True,
    #         )
    #         prediction = pred[0]

    #     # print("pred: " + str(pred))

    #     if (
    #         perturbed <= self.RL_TRAIN_BATCH[action[0 : len(action)]].max()
    #         and perturbed
    #         >= self.RL_TRAIN_BATCH[
    #             action[0 : len(action)]
    #         ].min()  # keep the semantic intact
    #         and prediction > 0.5 # pred[1] > 0.5 #for discriminator
    #     ):  # evasion happened
    #         self.reward = 1

    #         self.evasion_count += 1

    #         # ==================== Adversarial Training ==================================
    #         if ADVERSARIAL_TRAINING:

    #             batch_size = 1

    #             real = np.ones((batch_size, 1))
    #             bot_label = np.zeros((batch_size, 1))
    #             normal_label = np.ones((batch_size, 1))

    #             d_l_E = self.discriminator.train_on_batch(
    #                 np.array([self.observation_space]),
    #                 [real, bot_label, normal_label],
    #             )

    #     # ======================================================
    #     else:
    #         self.reward = 0

    #     self.sample_number += 1
    #     if self.sample_number >= self.RL_TRAIN_BATCH.shape[0] :  # If it's equal to 255 then exit
    #         episode_over = True
    #         self.sample_number = 0

    #     return self.observation_space, self.reward, episode_over, {}

    def step(self, action_index):

        episode_over = False

        if self.DATA_SET == "ISCX-2014":
            ACTION_LOOKUP = {
                i: act for i, act in enumerate(ACTION_TABLE_ISCX_2014.keys())
            }
        elif self.DATA_SET == "CIC-2017":
            ACTION_LOOKUP = {
                i: act for i, act in enumerate(ACTION_TABLE_CIC_2017.keys())
            }
        elif self.DATA_SET == "CIC-2018":
            ACTION_LOOKUP = {
                i: act for i, act in enumerate(ACTION_TABLE_CIC_2018.keys())
            }

        assert action_index < len(ACTION_LOOKUP)
        action = ACTION_LOOKUP[action_index]

        delta = self.RL_TRAIN_BATCH[action[0 : len(action)]].min()

        perturbed = (
            self.RL_TRAIN_BATCH.iloc[self.sample_number % self.RL_TRAIN_BATCH.shape[0]][
                action[0 : len(action)]
            ]
            + delta
        )

        self.temp_sample = self.RL_TRAIN_BATCH.copy()

        self.temp_sample.iloc[self.sample_number % self.temp_sample.shape[0]][
            action[0 : len(action) - 1]
        ] = perturbed  # %256 remiander, Assign new value to the selected feature

        if USE_RL_AT_THE_END == False:
            pred = self.discriminator.predict(
                np.array(
                    [
                        self.temp_sample.iloc[
                            self.sample_number % self.temp_sample.shape[0]
                        ],
                    ]
                )
            )
            prediction = pred[1]
        else:
            pred = predict_clf(
                np.array(
                    [
                        self.temp_sample.iloc[
                            self.sample_number % self.temp_sample.shape[0]
                        ],
                    ]
                ),
                np.array(
                    [
                        self.temp_sample.iloc[
                            self.sample_number % self.temp_sample.shape[0]
                        ],
                    ]
                ),
                np.array(
                    [
                        self.temp_sample.iloc[
                            self.sample_number % self.temp_sample.shape[0]
                        ],
                    ]
                ),
                self.XGB,
                ONLY_GZ=True,
            )
            prediction = pred[0]

        # print("pred: " + str(pred))

        if (
            perturbed <= self.RL_TRAIN_BATCH[action[0 : len(action)]].max()
            and perturbed
            >= self.RL_TRAIN_BATCH[
                action[0 : len(action)]
            ].min()  # keep the semantic intact
            and prediction > 0.5  # pred[1] > 0.5 #for discriminator
        ):  # evasion happened
            self.reward = 1

            self.evasion_count += 1

            episode_over = True
            if self.turns > 0:
                print("Attacks: " + str(self.turns))

            # ==================== Adversarial Training ==================================
            if ADVERSARIAL_TRAINING:

                batch_size = 1

                real = np.ones((batch_size, 1))
                bot_label = np.zeros((batch_size, 1))
                normal_label = np.ones((batch_size, 1))

                d_l_E = self.discriminator.train_on_batch(
                    np.array([self.observation_space]),
                    [real, bot_label, normal_label],
                )

        # ======================================================
        elif self.turns >= self.nb_actions:
            self.turns = 0
            episode_over = True
            self.reward = 0               
        
        else:
            self.reward = 0
            self.turns += 1
            # print("Missed")
            self.observation_space = self.temp_sample.iloc[
                self.sample_number % self.temp_sample.shape[0]
            ]

        

        return self.observation_space, self.reward, episode_over, {}

    def reset(self):
        if (
            self.sample_number >= self.RL_TRAIN_BATCH.shape[0]
        ):  # If it's equal to 255 then exit

            self.sample_number = -1
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>Done<<<<<<<<<<<<<<<<<<<<<<<<<")


        self.sample_number += 1

        self.observation_space = self.RL_TRAIN_BATCH.iloc[
            self.sample_number % self.RL_TRAIN_BATCH.shape[0]
        ]

        return self.observation_space

    def render(self, mode="human", close=False):
        pass

    # --------------------------------------------------------------------------------
    # ---------------------------RL Model --------------------------------------------
    # --------------------------------------------------------------------------------

    def generate_agent_model(self, input_shape, layers, nb_actions):
        model = Sequential()

        model.add(Flatten(input_shape=input_shape))
        model.add(
            Dropout(0.1)
        )  # drop out the input to make model less sensitive to any 1 feature

        for layer in layers:
            model.add(Dense(layer))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(nb_actions))
        model.add(Activation("linear"))

        return model

    def train_dqn_model(
        self, layers, rounds=256, run_test=False, use_score=False, env=0
    ):

        env.seed(123)

        if self.is_first_time:
            self.is_first_time = False

            self.nb_actions = env.action_space.n
            window_length = (
                1  # "experience" consists of where we were, where we are now
            )

            # print("self.nb_actions:" + str(self.nb_actions))
            # print('(window_length,) + env.observation_space.shape:' + str((window_length,) + env.observation_space.shape))
            # generate a policy model
            self.agent_nn = self.generate_agent_model(
                (window_length,) + env.observation_space.shape, layers, self.nb_actions
            )

            # configure and compile our agent
            # BoltzmannQPolicy selects an action stochastically with a probability generated by soft-maxing Q values
            policy = BoltzmannQPolicy()

            # memory can help a model during training
            # for this, we only consider a single malware sample (window_length=1) for each "experience"
            memory = SequentialMemory(
                limit=50000,
                ignore_episode_boundaries=False,
                window_length=window_length,
            )

            # DQN agent as described in Mnih (2013) and Mnih (2015).
            # http://arxiv.org/pdf/1312.5602.pdf
            # http://arxiv.org/abs/1509.06461
            self.agent = DQNAgent(
                model=self.agent_nn,
                nb_actions=self.nb_actions,
                memory=memory,
                nb_steps_warmup=100,
                enable_double_dqn=True,
                enable_dueling_network=True,
                dueling_type="avg",
                target_model_update=1e-3,
                policy=policy,
            )

            # self.agent = SarsaAgent(
            #     self.agent_nn,
            #     self.nb_actions,
            #     policy=policy,
            #     test_policy=policy,
            #     gamma=0.99,
            #     nb_steps_warmup=10,
            #     train_interval=1,
            #     delta_clip=inf,
            # )

            # keras-rl allows one to use and built-in keras optimizer
            self.agent.compile(Adam(lr=1e-3), metrics=["mae"])

        # play the game. learn something!

        self.agent.fit(env, nb_steps=rounds, visualize=False, verbose=1)

        # self.agent.test(env, nb_episodes=100, visualize=False)

        return self.agent

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    def build_generator(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.img_shape))
        model.add(Activation("relu"))
        print(model.metrics_names)
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype="int32")

        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = tf.keras.layers.multiply([noise, label_embedding])

        # model_input = multiply([noise, label_embedding])
        img = model(model_input)

        # model.summary()

        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()
        # model.add(Dense(256, input_dim=self.img_shape))
        # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Dropout(0.25))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        img = Input(shape=self.img_shape)
        # Extract feature representation
        features = model(img)
        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        bot = Dense(1, activation="sigmoid")(features)
        normal = Dense(1, activation="sigmoid")(features)

        # Dense(self.num_classes, activation="softmax")(features)
        # model.summary()
        return Model(img, [validity, bot, normal])

    def train(self, model_components):
        [
            cache_prefix,
            with_class,
            starting_step,
            train,
            Train,
            data_cols,
            data_dim,
            label_cols,
            label_dim,
            rand_noise_dim,
            nb_steps,
            batch_size,
            k_d,
            k_g,
            critic_pre_train_steps,
            log_interval,
            learning_rate,
            base_n_count,
            CACHE_PATH,
            FIGS_PATH,
            show,
            comb_loss,
            disc_loss_generated,
            disc_loss_real,
            xgb_losses,
            dt_losses,
            nb_losses,
            knn_losses,
            rf_losses,
            lr_losses,
            test_size,
            epoch_list_disc_loss_real,
            epoch_list_disc_loss_generated,
            epoch_list_comb_loss,
            gpu_device,
            EVALUATION_PARAMETER,
            TODAY,
            DATA_SET,
        ] = model_components
        batch_iteration = 0
        total_time_so_far = 0
        list_batch_iteration = []
        list_log_iteration = []
        list_reward = []
        list_evasions = []

        Disc_G_Z_Real_acc_ = []
        Disc_G_Z_N_acc_ = []
        Disc_G_Z_B_acc_ = []
        Disc_N_acc_ = []
        Disc_B_acc_ = []

        epoch_list_disc_loss_real = []
        epoch_list_disc_loss_generated = []
        epoch_list_gen_loss = []

        pred_G_Z_XGB = []
        pred_G_Z_DT = []
        pred_G_Z_NB = []
        pred_G_Z_RF = []
        pred_G_Z_LR = []
        pred_G_Z_KNN = []

        pred_total_XGB = []
        pred_total_DT = []
        pred_total_NB = []
        pred_total_RF = []
        pred_total_LR = []
        pred_total_KNN = []

        list_loss_real_bot = []
        list_loss_fake_bot = []
        list_loss_real_normal = []
        list_loss_g = []

        x = 0
        g_z = 0

        best_xgb_acc_index = 0
        best_xgb_rcl_index = 0
        best_dt_acc_index = 0
        best_dt_rcl_index = 0
        best_nb_acc_index = 0
        best_nb_rcl_index = 0
        best_rf_acc_index = 0
        best_rf_rcl_index = 0
        best_lr_acc_index = 0
        best_lr_rcl_index = 0
        best_knn_acc_index = 0
        best_knn_rcl_index = 0
        # Directory
        os.mkdir(CACHE_PATH + TODAY)
        os.mkdir(FIGS_PATH + TODAY)
        epoch_number = 0
        log_iteration = 0
        if DEBUG:
            print(cache_prefix, batch_size, base_n_count)
        i = 0

        self.is_first_time = True
        self.d_weights_updated = False
        self.cache_path = CACHE_PATH + TODAY + "/weights"
        self.DATA_SET = DATA_SET

        # --------------------------------------------------------------------------------------
        mean = 0
        stdv = 1
        # --------------------------------------------------------------------------------------

        if DEBUG:
            print("======================================================")
            print("Batch Size Selected -------->>>>>>> " + str(batch_size))
            print("======================================================")
        Bots = Train.loc[Train["Label"] == 0].copy()
        Normal = Train.loc[Train["Label"] == 1].copy()
        print("Normal: " + str(Normal.shape))
        print("Bots: " + str(Bots.shape))

        t_bots = Bots[0 : int(Bots.shape[0] * 0.7)]
        T_normal = Normal[0 : int(Normal.shape[0] * 0.7)]
        T = pd.concat([t_bots, T_normal]).reset_index(
            drop=True
        )  # Augmenting with real botnets
        # shuffled_T_B = T.sample(frac=1).reset_index(drop=True)
        # shuffled_Bots = shuffled_T_B.loc[shuffled_T_B["Label"] == 0].copy()
        # shuffled_Normal = shuffled_T_B.loc[shuffled_T_B["Label"] == 1].copy()
        # print(shuffled_Bots.shape, shuffled_Normal.shape)

        shuffled_T_B = T
        shuffled_Bots = t_bots.copy()
        shuffled_Normal = T_normal.copy()
        print(shuffled_Bots.shape, shuffled_Normal.shape)

        # print('self.RL_TRAIN_BATCH.shape[0]: ' + str(self.RL_TRAIN_BATCH.shape[0]))
        # shuffled_T_B.to_csv('shuffled_T_B.csv')
        # print('File: ' + 'shuffled_T_B.csv saved to directory')

        # print(shuffled_Normal.columns)

        test_Normal = Normal[int(Normal.shape[0] * 0.7) : Normal.shape[0]]
        test_Bots = Bots[int(Bots.shape[0] * 0.7) : Bots.shape[0]]

        if ESTIMATE_CLASSIFIERS:
            print("Estimating Classifiers..")
            self.XGB = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
            # DT = DecisionTreeClassifier()
            # NB = GaussianNB()
            # RF = RandomForestClassifier()
            # LR = LogisticRegression(max_iter=10000000)
            # KNN = KNeighborsClassifier()

            print("XGB..")
            self.XGB.fit(shuffled_T_B[data_cols], shuffled_T_B["Label"])

            # print("DT..")
            # DT.fit(shuffled_T_B[data_cols], shuffled_T_B["Label"])

            # print("NB..")
            # NB.fit(shuffled_T_B[data_cols], shuffled_T_B["Label"])

            # print("RF..")
            # RF.fit(shuffled_T_B[data_cols], shuffled_T_B["Label"])

            # print("LR..")
            # LR.fit(shuffled_T_B[data_cols], shuffled_T_B["Label"])

            # print("KNN..")
            # KNN.fit(shuffled_T_B[data_cols], shuffled_T_B["Label"])

            # G_Z = test_Normal.copy()

            # pred_total_XGB = predict_clf(
            #     G_Z[data_cols],
            #     test_Normal[data_cols],
            #     test_Bots[data_cols],
            #     XGB,
            #     ONLY_GZ=False,
            # )
            # pred_total_DT = predict_clf(
            #     G_Z[data_cols],
            #     test_Normal[data_cols],
            #     test_Bots[data_cols],
            #     DT,
            #     ONLY_GZ=False,
            # )
            # pred_total_NB = predict_clf(
            #     G_Z[data_cols],
            #     test_Normal[data_cols],
            #     test_Bots[data_cols],
            #     NB,
            #     ONLY_GZ=False,
            # )
            # pred_total_RF = predict_clf(
            #     G_Z[data_cols],
            #     test_Normal[data_cols],
            #     test_Bots[data_cols],
            #     RF,
            #     ONLY_GZ=False,
            # )
            # pred_total_LR = predict_clf(
            #     G_Z[data_cols],
            #     test_Normal[data_cols],
            #     test_Bots[data_cols],
            #     LR,
            #     ONLY_GZ=False,
            # )
            # pred_total_KNN = predict_clf(
            #     G_Z[data_cols],
            #     test_Normal[data_cols],
            #     test_Bots[data_cols],
            #     KNN,
            #     ONLY_GZ=False,
            # )

        print("======================================================")
        print("Starting GAN Training..")
        print("======================================================")

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        start_log_time = time.time()

        # self.reset()

        batch_num = 0

        self.rounds = batch_size

        env = RELEVAGAN_CC()

        for i in range(starting_step, starting_step + nb_steps):
            # real = np.random.uniform(low=0.999, high=1.0, size=batch_size)
            # fake = np.random.uniform(low=0, high=0.001, size=batch_size)
            # print("Training Disciminator----------------->>>\n")
            for j in range(k_d):
                np.random.seed(i + j)
                z = np.random.normal(mean, stdv, size=(batch_size, rand_noise_dim))
                # if USE_UNIFORM_NOISE:
                #     z = np.random.uniform(mean, stdv, size=(batch_size, rand_noise_dim))
                t_b = get_data_batch(
                    shuffled_Bots,
                    batch_size,
                    seed=i + j,
                )  # t_b is the set of Botnet traffic samples

                T_b = get_data_batch(
                    shuffled_Normal,
                    batch_size,
                    seed=i + j,
                )  # T_b is the set of Normal traffic samples
                labels = t_b["Label"]
                Labels = T_b["Label"]

                # LABELS = T_B['Label']
                # sampled_labels = np.random.randint(0, 1, (batch_size, 1))
                # print("Going to predict from G...")

                if (
                    self.minority_training_count
                    < shuffled_Bots.shape[0] // batch_size + 1
                ):
                    g_z = self.generator.predict([z, labels])

                    d_l_g = self.discriminator.train_on_batch(
                        g_z, [fake, labels, Labels]
                    )

                    d_l_B = self.discriminator.train_on_batch(
                        t_b[data_cols], [real, labels, Labels]
                    )

                    self.minority_training_count += 1

                    if USE_RL_AFTER_BATCH:

                        self.RL_TRAIN_BATCH = t_b[data_cols].copy()
                        # print('self.RL_TRAIN_BATCH.shape[0]: ' + str(self.RL_TRAIN_BATCH.shape[0]))

                        agent1 = self.train_dqn_model(
                            [256, 128, 64, 32],
                            rounds=2000,
                            run_test=False,
                            env=self,
                        )  # black blox

                        # agent_time_taken = time.time() - agent_time_startk

                        # print("Agent time taken:" + str(agent_time_taken))

                        # print("batch_reward:" + str(self.reward))
                        # model1.save('models/dqn.h5', overwrite=True)
                        # with open('history_blackbox.pickle', 'wb') as f:
                        # pickle.dump(history_test1, f, pickle.HIGHEST_PROTOCOL)

                        # self.ROUNDS_FACTOR += 1

                d_l_N = self.discriminator.train_on_batch(
                    T_b[data_cols], [real, labels, Labels]
                )

                # d_l_N= [2.7684414, 0.97742355, 0.88196146, 0.9090565]

                # --------------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------------

            for j in range(k_g):
                np.random.seed(i + j)
                z = np.random.normal(mean, stdv, size=(batch_size, rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(mean, stdv, size=(batch_size, rand_noise_dim))
                # g_loss = self.combined.train_on_batch([z, labels], [real,  Labels, labels])
                g_loss = self.combined.train_on_batch([z, labels], [real, labels])
                # print(g_loss)

            list_batch_iteration.append(batch_iteration)

            # print(
            #         "d_l_g: "
            #         + str(d_l_g[0])
            #         + "  d_l_N: "
            #         + str(d_l_N[0])
            #         + " d_l_B: "
            #         + str(d_l_B[0])
            #         + "  g_loss: "
            #         + str(g_loss[0])
            #     )

            # --------------------------------------------------------------------------------------

            # print('d_l_g: ' + str(d_l_g[0]) + '  d_l_N: ' + str(d_l_N[0]) + ' d_l_B: ' + str(d_l_B[0]) + '  g_loss: ' + str(g_loss[0]))
            # print('d_l_g: ' + str(d_l_g[0]) + '  d_l_N: ' + str(d_l_N[0]) + '  d_l_B: ' + str(d_l_B[0]) + '  g_loss: ' + str(g_loss[0]))

            # --------------------------------------------------------------------------------------
            # print('batch#: ' + str(batch_num)  + '/' + str(log_interval) + ' completed..')
            # batch_num = batch_num + 1
            # if batch_num >= log_interval: batch_num = 0

            # print("Evasions: " + str(self.evasion_count))
            # self.reward = 0
            # self.evasion_count = 0

            # Determine xgb loss each step, after training generator and discriminator
            if i % log_interval == 0:  # 2x faster than testing each step...

                # print(
                #     "d_l_g: "
                #     + str(d_l_g[0])
                #     + "  d_l_N: "
                #     + str(d_l_N[0])
                #     + " d_l_B: "
                #     + str(d_l_B[0])
                #     + "  g_loss: "
                #     + str(g_loss[0])
                # )

                print("  g_loss: " + str(g_loss[0]))
                if USE_RL_AFTER_EPOCH:

                    self.RL_TRAIN_BATCH = t_bots[data_cols].copy()
                    print(
                        "self.RL_TRAIN_BATCH.shape[0]: "
                        + str(self.RL_TRAIN_BATCH.shape[0])
                    )

                    agent1 = self.train_dqn_model(
                        [1024, 512],
                        rounds=self.RL_TRAIN_BATCH.shape[0] * 10,
                        run_test=False,
                        env=self,
                    )  # black blox

                    # agent_time_taken = time.time() - agent_time_start

                    # print("Agent time taken:" + str(agent_time_taken))

                    # print("batch_reward:" + str(self.reward))
                    # model1.save('models/dqn.h5', overwrite=True)
                    # with open('history_blackbox.pickle', 'wb') as f:
                    # pickle.dump(history_test1, f, pickle.HIGHEST_PROTOCOL)

                    # self.ROUNDS_FACTOR += 1
                print("Evasions: " + str(self.evasion_count))
                self.minority_training_count = 0

                list_reward.append(self.reward)
                list_evasions.append(self.evasion_count)

                self.evasion_count = 0
                # self.reward = 0

                list_log_iteration.append(log_iteration)
                log_iteration = log_iteration + 1

                epoch_list_disc_loss_real.append(d_l_B[0])  # epoch_list_disc_loss_real
                epoch_list_disc_loss_generated.append(
                    d_l_g[0]
                )  # epoch_list_disc_loss_gen
                epoch_list_gen_loss.append(g_loss[0])  # epoch_list_gen_loss

                list_loss_real_bot.append(d_l_B[0])
                list_loss_fake_bot.append(d_l_g[0])
                list_loss_real_normal.append(d_l_N[0])
                list_loss_g.append(g_loss[0])

                X_for_plot = get_data_batch(t_bots, t_bots.shape[0], seed=i)
                z = np.random.normal(size=(t_bots.shape[0], rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(size=(t_bots[0].shape, rand_noise_dim))
                labels = X_for_plot["Label"]
                G_Z_for_plot = self.generator.predict([z, labels])
                # g_z -= g_z.min()
                # g_z /= g_z.max()
                X_for_plot = X_for_plot[data_cols]
                # pred = self.discriminator.predict(Test[data_cols]).copy()
                # ====================Testing on 30% of train data ================================

                z = np.random.normal(size=(test_Bots.shape[0], rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(size=(test_Bots.shape[0], rand_noise_dim))
                G_z = self.generator.predict([z, test_Bots["Label"]])

                # print(test_Bots["Label"])

                G_Z = pd.DataFrame(G_z).copy()
                # print("G_Z.shape: " + str(G_Z.shape))
                G_Z.columns = data_cols
                G_Z["Label"] = 0
                pred_G_Z = self.discriminator.predict(G_Z[data_cols])
                pred_Normal = self.discriminator.predict(test_Normal[data_cols])
                pred_Bots = self.discriminator.predict(test_Bots[data_cols])
                G_Z_Real_acc_ = round(sum(pred_G_Z[0]) / G_Z.shape[0], 4)
                Ev_GZ_Bot = round(
                    sum(pred_G_Z[1]) / G_Z.shape[0], 4
                )  # predict generated bot being real. If it maintains near 1 then it means it is bot.
                N_acc_ = round(
                    sum(pred_Normal[2]) / test_Normal.shape[0], 4
                )  # predict normal being normal. If it maintains near 1 then it means it is normal becasue normal has been labeled as 0.
                Ev_Real_Bot = round(
                    sum(pred_Bots[1]) / test_Bots.shape[0], 4
                )  # predict bot being bot. If it maintains near 0 then it means it is bot because bot has been labeled as 1.
                Disc_G_Z_Real_acc_ = np.append(Disc_G_Z_Real_acc_, G_Z_Real_acc_)
                Disc_G_Z_B_acc_ = np.append(Disc_G_Z_B_acc_, Ev_GZ_Bot)
                Disc_N_acc_ = np.append(Disc_N_acc_, N_acc_)
                Disc_B_acc_ = np.append(Disc_B_acc_, Ev_Real_Bot)

                # print("++++++++++++++++++++++++")
                # print("GEN_VALIDITY: " + str(Disc_G_Z_Real_acc_))
                # print("FAKE_BOT_EVA: " + str(Disc_G_Z_B_acc_))
                # print("REAL_NORMAL_EST: " + str(Disc_N_acc_))
                # print("REAL_BOT_EVA: " + str(Disc_B_acc_))
                # print("++++++++++++++++++++++++")

                # ---------------------------------------------------------
                if SHOW_TIME:
                    end_log_time = time.time()
                    log_interval_time = end_log_time - start_log_time
                    start_log_time = time.time()
                    total_time_so_far += log_interval_time
                    if DEBUG:
                        print(
                            "log_iteration: "
                            + str(log_iteration)
                            + "/"
                            + str(nb_steps // log_interval)
                        )
                    # print("Time taken so far: " + str(total_time_so_far)  + " seconds")
                    total_time = (
                        total_time_so_far / log_iteration * nb_steps // log_interval
                    )
                    if DEBUG:
                        print(
                            "Average time per log_iteration: "
                            + str(total_time_so_far / log_iteration)
                        )
                    time_left = round((total_time - total_time_so_far) / 3600, 2)
                    time_unit = "hours"
                    if time_left < 1:
                        time_left = round(time_left * 60, 2)
                        time_unit = "minutes"
                    print("Time left = " + str(time_left) + " " + time_unit)
                    print(
                        "Total Time Taken: "
                        + str(round(total_time_so_far / 60, 1))
                        + " minutes"
                    )

                # save model checkpoints
                # model_checkpoint_base_name = CACHE_PATH + TODAY + \
                #     '/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                # self.generator.save_weights(
                #     model_checkpoint_base_name.format('self.generator',
                #                                       epoch_number))
                # self.discriminator.save_weights(
                #     model_checkpoint_base_name.format('discriminator',
                #                                       epoch_number))

                epoch_number = epoch_number + 1

                print("epoch_number: " + str(epoch_number) + " completed")
                print("======================================================")

                if epoch_number % 10 == 0:

                    # PlotData(
                    #     X_for_plot,
                    #     G_Z_for_plot,
                    #     data_cols=data_cols,
                    #     label_cols=label_cols,
                    #     seed=0,
                    #     with_class=with_class,
                    #     data_dim=data_dim,
                    #     save=False,
                    #     list_batch_iteration=list_batch_iteration,
                    #     Disc_G_Z_Real_acc_=Disc_G_Z_Real_acc_,
                    #     Disc_G_Z_N_acc_=Disc_G_Z_N_acc_,
                    #     Disc_G_Z_B_acc_=Disc_G_Z_B_acc_,
                    #     Disc_N_acc_=Disc_N_acc_,
                    #     Disc_B_acc_=Disc_B_acc_,
                    #     epoch_list_disc_loss_real=epoch_list_disc_loss_real,
                    #     epoch_list_disc_loss_generated=epoch_list_disc_loss_generated,
                    #     epoch_list_gen_loss=epoch_list_gen_loss,
                    #     list_log_iteration=list_log_iteration,
                    #     figs_path=FIGS_PATH,
                    #     cache_prefix=cache_prefix,
                    #     save_fig=False,
                    #     GAN_type="ACGAN",
                    #     TODAY=TODAY,
                    # )

                    GEN_VALIDITY = pd.DataFrame(
                        Disc_G_Z_Real_acc_, columns=["GEN_VALIDITY"]
                    )
                    FAKE_BOT_EVA = pd.DataFrame(
                        Disc_G_Z_B_acc_, columns=["FAKE_BOT_EVA"]
                    )
                    REAL_NORMAL_EST = pd.DataFrame(
                        Disc_N_acc_, columns=["REAL_NORMAL_EST"]
                    )
                    REAL_BOT_EVA = pd.DataFrame(Disc_B_acc_, columns=["REAL_BOT_EVA"])

                    D_Loss_Real_Bot = pd.DataFrame(
                        list_loss_real_bot, columns=["D_Loss_Real_Bot"]
                    )
                    D_Loss_Fake_Bot = pd.DataFrame(
                        list_loss_fake_bot, columns=["D_Loss_Fake_Bot"]
                    )
                    D_Loss_Real_Normal = pd.DataFrame(
                        list_loss_real_normal, columns=["D_Loss_Real_Normal"]
                    )
                    G_Loss = pd.DataFrame(list_loss_g, columns=["G_Loss"])

                    rewards = pd.DataFrame(list_reward, columns=["reward"])
                    evasions = pd.DataFrame(list_evasions, columns=["evasions"])

                    time_taken = pd.DataFrame(
                        [round(total_time_so_far / 60, 1)], columns=["Time"]
                    )

                    if POST_TRAINING_ESTIMATE_CLASSIFIERS:

                        pred_G_Z_XGB = predict_clf(
                            G_Z[data_cols],
                            test_Normal[data_cols],
                            test_Bots[data_cols],
                            XGB,
                            ONLY_GZ=True,
                        )
                        pred_G_Z_DT = predict_clf(
                            G_Z[data_cols],
                            test_Normal[data_cols],
                            test_Bots[data_cols],
                            DT,
                            ONLY_GZ=True,
                        )
                        pred_G_Z_NB = predict_clf(
                            G_Z[data_cols],
                            test_Normal[data_cols],
                            test_Bots[data_cols],
                            NB,
                            ONLY_GZ=True,
                        )
                        pred_G_Z_RF = predict_clf(
                            G_Z[data_cols],
                            test_Normal[data_cols],
                            test_Bots[data_cols],
                            RF,
                            ONLY_GZ=True,
                        )
                        pred_G_Z_LR = predict_clf(
                            G_Z[data_cols],
                            test_Normal[data_cols],
                            test_Bots[data_cols],
                            LR,
                            ONLY_GZ=True,
                        )
                        pred_G_Z_KNN = predict_clf(
                            G_Z[data_cols],
                            test_Normal[data_cols],
                            test_Bots[data_cols],
                            KNN,
                            ONLY_GZ=True,
                        )

                        pred_total_XGB = pd.DataFrame(pred_total_XGB, columns=["XGB"])
                        pred_total_DT = pd.DataFrame(pred_total_DT, columns=["DT"])
                        pred_total_NB = pd.DataFrame(pred_total_NB, columns=["NB"])
                        pred_total_RF = pd.DataFrame(pred_total_RF, columns=["RF"])
                        pred_total_LR = pd.DataFrame(pred_total_LR, columns=["LR"])
                        pred_total_KNN = pd.DataFrame(pred_total_KNN, columns=["KNN"])

                        pred_G_Z_XGB = pd.DataFrame(
                            pred_G_Z_XGB, columns=["pred_G_Z_XGB"]
                        )
                        pred_G_Z_DT = pd.DataFrame(pred_G_Z_DT, columns=["pred_G_Z_DT"])
                        pred_G_Z_NB = pd.DataFrame(pred_G_Z_NB, columns=["pred_G_Z_NB"])
                        pred_G_Z_RF = pd.DataFrame(pred_G_Z_RF, columns=["pred_G_Z_RF"])
                        pred_G_Z_LR = pd.DataFrame(pred_G_Z_LR, columns=["pred_G_Z_LR"])
                        pred_G_Z_KNN = pd.DataFrame(
                            pred_G_Z_KNN, columns=["pred_G_Z_KNN"]
                        )

                        frames = [
                            GEN_VALIDITY,
                            FAKE_BOT_EVA,
                            REAL_NORMAL_EST,
                            REAL_BOT_EVA,
                            D_Loss_Real_Bot,
                            D_Loss_Fake_Bot,
                            D_Loss_Real_Normal,
                            G_Loss,
                            rewards,
                            evasions,
                            pred_total_XGB,
                            pred_total_DT,
                            pred_total_NB,
                            pred_total_RF,
                            pred_total_LR,
                            pred_total_KNN,
                            pred_G_Z_XGB,
                            pred_G_Z_DT,
                            pred_G_Z_NB,
                            pred_G_Z_RF,
                            pred_G_Z_LR,
                            pred_G_Z_KNN,
                            time_taken,
                        ]
                    else:
                        frames = [
                            GEN_VALIDITY,
                            FAKE_BOT_EVA,
                            REAL_NORMAL_EST,
                            REAL_BOT_EVA,
                            D_Loss_Real_Bot,
                            D_Loss_Fake_Bot,
                            D_Loss_Real_Normal,
                            G_Loss,
                            rewards,
                            evasions,
                            time_taken,
                        ]

                    LISTS = pd.concat(frames, sort=True, axis=1).to_csv(
                        CACHE_PATH
                        + TODAY
                        + "/"
                        + str(epoch_number)
                        + "RELEVAGAN_CC_LISTS.csv"
                    )
            USE_RL_AT_THE_END = True

        if USE_RL_AT_THE_END:

            print("here>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            self.RL_TRAIN_BATCH = t_b[data_cols].copy()
            # print('self.RL_TRAIN_BATCH.shape[0]: ' + str(self.RL_TRAIN_BATCH.shape[0]))

            agent1 = self.train_dqn_model(
                [1024, 512],
                rounds=self.RL_TRAIN_BATCH.shape[0] * 100,
                run_test=False,
                env=self,
            )  # black blox

            # agent_time_taken = time.time() - agent_time_start

            # print("Agent time taken:" + str(agent_time_taken))

            # print("batch_reward:" + str(self.reward))
            # model1.save('models/dqn.h5', overwrite=True)
            # with open('history_blackbox.pickle', 'wb') as f:
            # pickle.dump(history_test1, f, pickle.HIGHEST_PROTOCOL)

            # self.ROUNDS_FACTOR += 1

        epoch_number = 0
        log_iteration = 0
        epoch_number = 0

        return [
            best_xgb_acc_index,
            best_xgb_rcl_index,
            best_dt_acc_index,
            best_dt_rcl_index,
            best_nb_acc_index,
            best_nb_rcl_index,
            best_rf_acc_index,
            best_rf_rcl_index,
            best_lr_acc_index,
            best_lr_rcl_index,
            best_knn_acc_index,
            best_knn_rcl_index,
        ]


def train_RELEVAGAN_CC(
    arguments, train, Train, data_cols, label_cols=[], seed=0, starting_step=0
):
    [
        rand_noise_dim,
        nb_steps,
        batch_size,
        k_d,
        k_g,
        critic_pre_train_steps,
        log_interval,
        learning_rate,
        base_n_count,
        CACHE_PATH,
        FIGS_PATH,
        show,
        test_size,
        gpu_device,
        EVALUATION_PARAMETER,
        TODAY,
        DATA_SET,
    ] = arguments
    with_class = False
    # np.random.seed(seed)     # set random seed
    data_dim = len(data_cols)

    # print('data_dim: ', data_dim)
    # print('data_cols: ', data_cols)
    label_dim = 0
    cache_prefix = "RELEVAGAN_CC"
    (
        comb_loss,
        disc_loss_generated,
        disc_loss_real,
        xgb_losses,
        dt_losses,
        nb_losses,
        knn_losses,
        rf_losses,
        lr_losses,
        epoch_list_disc_loss_real,
        epoch_list_disc_loss_generated,
        epoch_list_comb_loss,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])
    model_components = [
        cache_prefix,
        with_class,
        starting_step,
        train,
        Train,
        data_cols,
        data_dim,
        label_cols,
        label_dim,
        rand_noise_dim,
        nb_steps,
        batch_size,
        k_d,
        k_g,
        critic_pre_train_steps,
        log_interval,
        learning_rate,
        base_n_count,
        CACHE_PATH,
        FIGS_PATH,
        show,
        comb_loss,
        disc_loss_generated,
        disc_loss_real,
        xgb_losses,
        dt_losses,
        nb_losses,
        knn_losses,
        rf_losses,
        lr_losses,
        test_size,
        epoch_list_disc_loss_real,
        epoch_list_disc_loss_generated,
        epoch_list_comb_loss,
        gpu_device,
        EVALUATION_PARAMETER,
        TODAY,
        DATA_SET,
    ]
    [
        best_xgb_acc_index,
        best_xgb_rcl_index,
        best_dt_acc_index,
        best_dt_rcl_index,
        best_nb_acc_index,
        best_nb_rcl_index,
        best_rf_acc_index,
        best_rf_rcl_index,
        best_lr_acc_index,
        best_lr_rcl_index,
        best_knn_acc_index,
        best_knn_rcl_index,
    ] = RELEVAGAN_CC(IMG_SHAPE=len(data_cols)).train(model_components)
    return [
        best_xgb_acc_index,
        best_xgb_rcl_index,
        best_dt_acc_index,
        best_dt_rcl_index,
        best_nb_acc_index,
        best_nb_rcl_index,
        best_rf_acc_index,
        best_rf_rcl_index,
        best_lr_acc_index,
        best_lr_rcl_index,
        best_knn_acc_index,
        best_knn_rcl_index,
    ]

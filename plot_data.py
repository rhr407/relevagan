import header
import importlib

importlib.reload(header)  # For reloading after making changes
from header import *

DEBUG = 0
SHOW_KDE = 1
SHOW_CLASSIFIERS = 0
SHOW_EVAGAN = 1

SHOW_ACGAN = 1


SHOW_PLOTS = 1


def PlotData(
    x,
    g_z,
    data_cols=[],
    label_cols=[],
    seed=0,
    with_class=[],
    data_dim=[],
    save=False,
    list_batch_iteration=[],
    Disc_G_Z_Real_acc_=[],
    Disc_G_Z_N_acc_=[],
    Disc_G_Z_B_acc_=[],
    Disc_N_acc_=[],
    Disc_B_acc_=[],

    xgb_acc=[],
    dt_acc=[],
    nb_acc=[],
    knn_acc=[],
    rf_acc=[],
    lr_acc=[],
    xgb_rcl=[],
    dt_rcl=[],
    nb_rcl=[],
    knn_rcl=[],
    rf_rcl=[],
    lr_rcl=[],

    epoch_list_disc_loss_real=[],
    epoch_list_disc_loss_generated=[],
    epoch_list_gen_loss=[],
    list_log_iteration=[],
    figs_path="",
    cache_prefix="",
    save_fig=False,
    GAN_type="",
    TODAY="",
):

    # ===================================== Show KDE for a single feature =========================================================

    # if SHOW_KDE:

    # 	plt.style.use('seaborn-white')

    # 	real_bots_df = real_samples[data_cols].copy()
    # 	real_bots_df['Type'] = 'Real Bots'

    # 	print(real_bots_df.shape)

    # 	gan_bots_df = gen_samples[data_cols].copy()
    # 	gan_bots_df['Type'] = 'GAN Bots'

    # 	print(gan_bots_df.shape)

    # 	data_frame = gan_bots_df.copy()
    # 	data_frame_of_bots = pd.concat([real_bots_df, data_frame]).reset_index(drop=True).copy()

    # 	fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

    # 	for i in range(0, len(data_cols) - 1, 1):

    # 		sns.kdeplot(data = data_frame_of_bots, x = data_cols[i], y = data_cols[i+1],  bw_adjust=2, fill = False, hue = 'Type')
    # 		# sns.kdeplot(data = real_bots_df, x = data_cols[i], fill = True, alpha = 0.5, linewidth = 0)
    # 		# sns.kdeplot(data = gan_bots_df, x = data_cols[i], fill = True, alpha = 0.5, linewidth = 0)

    # 		# if SHOW_PLOTS:
    # 		plt.show()

    # 		plt.close()

    if SHOW_KDE:

        if GAN_type == "GAN":

            real_samples = pd.DataFrame(x, columns=data_cols)
            gen_samples = pd.DataFrame(g_z, columns=data_cols)

        elif GAN_type == "ACGAN":

            real_samples = pd.DataFrame(x, columns=data_cols + label_cols)
            gen_samples = pd.DataFrame(g_z, columns=data_cols + label_cols)

        else:
            real_samples = pd.DataFrame(x)
            gen_samples = pd.DataFrame(g_z)

        if DEBUG:
            # print("Shape of real Samples is =" + str(real_samples.shape))
            print("Shape of gen Samples is =" + str(gen_samples.shape))

            # print(real_samples.describe())
            print(gen_samples.describe())

        plt.style.use("seaborn-white")

        real_bots_df = real_samples[data_cols].copy()
        real_bots_df["Type"] = "Real Bots"

        print(real_bots_df.shape)

        gan_bots_df = gen_samples[data_cols].copy()
        gan_bots_df["Type"] = "GAN Bots"

        print('real_bots_df.shape: ' + str(real_bots_df.shape))

        print('gan_bots_df.shape: ' + str(gan_bots_df.shape))

        data_frame = gan_bots_df.copy()
        data_frame_of_bots = (
            pd.concat([real_bots_df, data_frame]).reset_index(drop=True).copy()
        )

        fig, axes = plt.subplots(9, 8, figsize=(18, 18))

        fig.tight_layout(h_pad=2.5, w_pad=2.5)

        j = 1
        k = 0

        g = sns.scatterplot(
            data=data_frame_of_bots,
            x=data_cols[0],
            y=data_cols[1],
            hue="Type",
            style="Type",
            ax=axes[0, 0],
            size="Type",
            sizes=(30, 40),
            alpha=0.3,
        )

        # g.set(ylim=(0, 1), xlim=(0, 1))

        for i in range(1, len(data_cols) - 1, 1):

            g = sns.scatterplot(
                data=data_frame_of_bots,
                x=data_cols[i],
                y=data_cols[i + 1],
                hue="Type",
                style="Type",
                ax=axes[k, j],
                legend=False,
                size="Type",
                sizes=(30, 40),
                alpha=0.3,
            )

            j = j + 1

            if j > 7:
                j = 0
                k = k + 1

            # g.set(ylim=(0, 1), xlim=(0, 1))

        # if SHOW_PLOTS:
        plt.show()

        plt.close()

    # ==================================Best Features for ISCX-2014 Dataset================================================================================

    # fig, axes = plt.subplots(2, 8, figsize=(30, 3))

    # fig.tight_layout(h_pad=2.5, w_pad = 2.5)

    # g = sns.scatterplot(data=data_frame_of_bots, x= data_cols[8], y=data_cols[9], hue="Type", style="Type", ax=axes[0, 0])
    # g = sns.scatterplot(data=data_frame_of_bots, x= data_cols[9], y=data_cols[10], hue="Type", style="Type", ax=axes[0, 1], legend = False)
    # g = sns.scatterplot(data=data_frame_of_bots, x= data_cols[12], y=data_cols[13], hue="Type", style="Type", ax=axes[0, 2], legend = False)
    # g = sns.scatterplot(data=data_frame_of_bots, x= data_cols[34], y=data_cols[35], hue="Type", style="Type", ax=axes[0, 3], legend = False)
    # g = sns.scatterplot(data=data_frame_of_bots, x= data_cols[35], y=data_cols[36], hue="Type", style="Type", ax=axes[0, 4], legend = False)
    # g = sns.scatterplot(data=data_frame_of_bots, x= data_cols[44], y=data_cols[45], hue="Type", style="Type", ax=axes[0, 5], legend = False)
    # g = sns.scatterplot(data=data_frame_of_bots, x= data_cols[45], y=data_cols[46], hue="Type", style="Type", ax=axes[0, 6], legend = False)
    # g = sns.scatterplot(data=data_frame_of_bots, x= data_cols[51], y=data_cols[52], hue="Type", style="Type", ax=axes[0, 7], legend = False)

    # g.set(ylim=(0, 1), xlim=(0, 1))

    # plt.show()

    # plt.close()

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    if GAN_type == "EVAGAN_CV":

        f, ax = plt.subplots(1, 1)

        ax.set_xlabel("Epochs")  # Add x label to all plots
        ax.set_ylabel("Prediction")  # Add x label to all plots

        best_Disc_G_Z_Real_acc_ = list(Disc_G_Z_Real_acc_).index(
            Disc_G_Z_Real_acc_.max()
        )
        ax.plot(
            list_log_iteration,
            Disc_G_Z_Real_acc_,
            color="magenta",
            label="Est_(G_Z)_REAL(max, epoch) = ({}, {})".format(
                round(Disc_G_Z_Real_acc_.max(), 4), best_Disc_G_Z_Real_acc_
            ),
        )
        ax.axvline(x=best_Disc_G_Z_Real_acc_, color="magenta")

        best_Disc_N_acc_ = list(Disc_N_acc_).index(Disc_N_acc_.max())
        ax.plot(
            list_log_iteration,
            Disc_N_acc_,
            color="green",
            label="Est_(N)_N(max, epoch)  = ({}, {})".format(
                round(Disc_N_acc_.max(), 4), best_Disc_N_acc_
            ),
        )
        ax.axvline(x=best_Disc_N_acc_, color="green")

        best_Disc_G_Z_B_acc_ = list(Disc_G_Z_B_acc_).index(Disc_G_Z_B_acc_.min())
        ax.plot(
            list_log_iteration,
            Disc_G_Z_B_acc_,
            color="blue",
            label="Eva(G_Z)(min, epoch) = ({}, {})".format(
                round(Disc_G_Z_B_acc_.min(), 4), best_Disc_G_Z_B_acc_
            ),
        )
        ax.axvline(x=best_Disc_G_Z_B_acc_, color="blue")

        best_disc_B_acc_ = list(Disc_B_acc_).index(Disc_B_acc_.min())
        ax.plot(
            list_log_iteration,
            Disc_B_acc_,
            color="red",
            label="Eva_(B)(min, epoch) = ({}, {})".format(
                round(Disc_B_acc_.min(), 4), best_disc_B_acc_
            ),
        )
        ax.axvline(x=best_disc_B_acc_, color="red")

        plt.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=1,
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()

        if save_fig:

            plt.savefig(
                figs_path
                + TODAY
                + "/"
                + cache_prefix
                + "_eval-Accuracy"
                + str(list_log_iteration[-1])
                + ".pdf",
                dpi=600,
            )

        if SHOW_PLOTS:
            plt.show()

        plt.close(f)

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    if GAN_type == "ACGAN_CV":

        f, ax = plt.subplots(1, 1)

        ax.set_xlabel("Epochs")  # Add x label to all plots
        ax.set_ylabel("Prediction")  # Add x label to all plots

        best_Disc_G_Z_Real_acc_ = list(Disc_G_Z_Real_acc_).index(
            Disc_G_Z_Real_acc_.max()
        )
        ax.plot(
            list_log_iteration,
            Disc_G_Z_Real_acc_,
            color="magenta",
            label="Est_(G_Z)_REAL(max, epoch) = ({}, {})".format(
                round(Disc_G_Z_Real_acc_.max(), 4), best_Disc_G_Z_Real_acc_
            ),
        )
        ax.axvline(x=best_Disc_G_Z_Real_acc_, color="magenta")

        best_Disc_N_acc_ = list(Disc_N_acc_).index(Disc_N_acc_.max())
        ax.plot(
            list_log_iteration,
            Disc_N_acc_,
            color="green",
            label="Est_(N)_N(max, epoch)  = ({}, {})".format(
                round(Disc_N_acc_.max(), 4), best_Disc_N_acc_
            ),
        )
        ax.axvline(x=best_Disc_N_acc_, color="green")

        # best_Disc_G_Z_B_acc_ = list(Disc_G_Z_B_acc_).index( Disc_G_Z_B_acc_.min())
        # ax.plot(list_log_iteration, Disc_G_Z_B_acc_ , color='blue', label='Ev(G_Z)(min, epoch) = ({}, {})'.format(round(Disc_G_Z_B_acc_.min(),4), best_Disc_G_Z_B_acc_))
        # ax.axvline(x=best_Disc_G_Z_B_acc_, color='blue')

        best_disc_B_acc_ = list(Disc_B_acc_).index(Disc_B_acc_.min())
        ax.plot(
            list_log_iteration,
            Disc_B_acc_,
            color="red",
            label="Eva_(B)(min, epoch) = ({}, {})".format(
                round(Disc_B_acc_.min(), 4), best_disc_B_acc_
            ),
        )
        ax.axvline(x=best_disc_B_acc_, color="red")

        plt.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=1,
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()

        if save_fig:

            plt.savefig(
                figs_path
                + TODAY
                + "/"
                + cache_prefix
                + "_eval-Accuracy"
                + str(list_log_iteration[-1])
                + ".pdf",
                dpi=600,
            )

        if SHOW_PLOTS:
            plt.show()

        plt.close(f)

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    if GAN_type == "ACGAN_CC":

        f, ax = plt.subplots(1, 1)

        ax.set_xlabel("Epochs")  # Add x label to all plots
        ax.set_ylabel("Prediction")  # Add x label to all plots

        best_Disc_G_Z_Real_acc_ = list(Disc_G_Z_Real_acc_).index(
            Disc_G_Z_Real_acc_.max()
        )
        ax.plot(
            list_log_iteration,
            Disc_G_Z_Real_acc_,
            color="magenta",
            label="Est_(G_Z)_REAL(max, epoch) = ({}, {})".format(
                round(Disc_G_Z_Real_acc_.max(), 4), best_Disc_G_Z_Real_acc_
            ),
        )
        ax.axvline(x=best_Disc_G_Z_Real_acc_, color="magenta")

        best_Disc_N_acc_ = list(Disc_N_acc_).index(Disc_N_acc_.max())
        ax.plot(
            list_log_iteration,
            Disc_N_acc_,
            color="green",
            label="Est_(N)_N(max, epoch)  = ({}, {})".format(
                round(Disc_N_acc_.max(), 4), best_Disc_N_acc_
            ),
        )
        ax.axvline(x=best_Disc_N_acc_, color="green")

        # best_Disc_G_Z_B_acc_ = list(Disc_G_Z_B_acc_).index( Disc_G_Z_B_acc_.min())
        # ax.plot(list_log_iteration, Disc_G_Z_B_acc_ , color='blue', label='Ev(G_Z)(min, epoch) = ({}, {})'.format(round(Disc_G_Z_B_acc_.min(),4), best_Disc_G_Z_B_acc_))
        # ax.axvline(x=best_Disc_G_Z_B_acc_, color='blue')

        best_disc_B_acc_ = list(Disc_B_acc_).index(Disc_B_acc_.min())
        ax.plot(
            list_log_iteration,
            Disc_B_acc_,
            color="red",
            label="Eva_(B)(min, epoch) = ({}, {})".format(
                round(Disc_B_acc_.min(), 4), best_disc_B_acc_
            ),
        )
        ax.axvline(x=best_disc_B_acc_, color="red")

        plt.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=1,
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()

        if save_fig:

            plt.savefig(
                figs_path
                + TODAY
                + "/"
                + cache_prefix
                + "_eval-Accuracy"
                + str(list_log_iteration[-1])
                + ".pdf",
                dpi=600,
            )

        if SHOW_PLOTS:
            plt.show()

        plt.close(f)

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    if SHOW_CLASSIFIERS:

        f, ax = plt.subplots(1, 1)

        ax.set_xlabel("Epochs")  # Add x label to all plots
        ax.set_ylabel("Accuracy")  # Add x label to all plots

        best_xgb_acc = list(xgb_acc).index(xgb_acc.min())
        ax.plot(
            list_log_iteration,
            xgb_acc,
            color="blue",
            label="XGB(min, epoch) = ({}, {})".format(
                round(xgb_acc.min(), 4), best_xgb_acc
            ),
        )
        ax.axvline(x=best_xgb_acc, color="blue")
        # ax.plot(list_log_iteration, xgb_acc , color='blue', marker=2, label='XGB(min, epoch) = ({}, {})'.format(round(xgb_acc.min(),4), best_xgb))

        if ALL_CLASSIFIERS:

            best_dt_acc = list(dt_acc).index(dt_acc.min())
            best_nb_acc = list(nb_acc).index(nb_acc.min())
            best_rf_acc = list(rf_acc).index(rf_acc.min())
            best_lr_acc = list(lr_acc).index(lr_acc.min())
            best_knn_acc = list(knn_acc).index(knn_acc.min())

            ax.plot(
                list_log_iteration,
                dt_acc,
                color="magenta",
                label="DT(min, epoch) = ({}, {})".format(
                    round(dt_acc.min(), 4), best_dt_acc
                ),
            )
            ax.plot(
                list_log_iteration,
                nb_acc,
                color="orange",
                label="NB(min, epoch) = ({}, {})".format(
                    round(nb_acc.min(), 4), best_nb_acc
                ),
            )
            ax.plot(
                list_log_iteration,
                rf_acc,
                color="green",
                label="RF(min, epoch) = ({}, {})".format(
                    round(rf_acc.min(), 4), best_rf_acc
                ),
            )
            ax.plot(
                list_log_iteration,
                lr_acc,
                color="magenta",
                label="LR(min, epoch) = ({}, {})".format(
                    round(lr_acc.min(), 4), best_lr_acc
                ),
            )
            ax.plot(
                list_log_iteration,
                knn_acc,
                color="purple",
                label="KNN(min, epoch) = ({}, {})".format(
                    round(knn_acc.min(), 4), best_knn_acc
                ),
            )

            ax.axvline(x=best_dt_acc, color="magenta")
            ax.axvline(x=best_nb_acc, color="orange")
            ax.axvline(x=best_rf_acc, color="green")
            ax.axvline(x=best_lr_acc, color="magenta")
            ax.axvline(x=best_knn_acc, color="purple")

        plt.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()

        if save_fig:

            plt.savefig(
                figs_path
                + TODAY
                + "/"
                + cache_prefix
                + "_eval-Accuracy"
                + str(list_log_iteration[-1])
                + ".pdf",
                dpi=600,
            )

        if SHOW_PLOTS:
            plt.show()

        plt.close(f)

        # =======================================================================================================================================
        # =======================================================================================================================================
        # =======================================================================================================================================

        # f, ax = plt.subplots(1, 1, figsize=(5, 3))
        f, ax = plt.subplots(1, 1)

        ax.set_xlabel("Epochs")  # Add x label to all plots
        ax.set_ylabel("Recall")  # Add x label to all plots

        best_xgb_rcl = list(xgb_rcl).index(xgb_rcl.min())

        ax.plot(
            list_log_iteration,
            xgb_rcl,
            color="blue",
            label="XGB(min, epoch) = ({}, {})".format(
                round(xgb_rcl.min(), 4), best_xgb_rcl
            ),
        )
        ax.axvline(x=best_xgb_rcl, color="blue")
        # ax.plot(list_log_iteration, xgb_rcl , color='blue', marker=2, label='XGB(min, epoch) = ({}, {})'.format(round(xgb_rcl.min(),4), best_xgb))

        if ALL_CLASSIFIERS:

            best_dt_rcl = list(dt_rcl).index(dt_rcl.min())
            best_nb_rcl = list(nb_rcl).index(nb_rcl.min())
            best_rf_rcl = list(rf_rcl).index(rf_rcl.min())
            best_lr_rcl = list(lr_rcl).index(lr_rcl.min())
            best_knn_rcl = list(knn_rcl).index(knn_rcl.min())

            ax.plot(
                list_log_iteration,
                dt_rcl,
                color="magenta",
                label="DT(min, epoch) = ({}, {})".format(
                    round(dt_rcl.min(), 4), best_dt_rcl
                ),
            )
            ax.plot(
                list_log_iteration,
                nb_rcl,
                color="orange",
                label="NB(min, epoch) = ({}, {})".format(
                    round(nb_rcl.min(), 4), best_nb_rcl
                ),
            )
            ax.plot(
                list_log_iteration,
                rf_rcl,
                color="green",
                label="RF(min, epoch) = ({}, {})".format(
                    round(rf_rcl.min(), 4), best_rf_rcl
                ),
            )
            ax.plot(
                list_log_iteration,
                lr_rcl,
                color="magenta",
                label="LR(min, epoch) = ({}, {})".format(
                    round(lr_rcl.min(), 4), best_lr_rcl
                ),
            )
            ax.plot(
                list_log_iteration,
                knn_rcl,
                color="purple",
                label="KNN(min, epoch) = ({}, {})".format(
                    round(knn_rcl.min(), 4), best_knn_rcl
                ),
            )

            ax.axvline(x=best_dt_rcl, color="magenta")
            ax.axvline(x=best_nb_rcl, color="orange")
            ax.axvline(x=best_rf_rcl, color="green")
            ax.axvline(x=best_lr_rcl, color="magenta")
            ax.axvline(x=best_knn_rcl, color="purple")

        # best_svm = list(svm_losses).index( svm_losses.min())
        # plt.axvline(x=best_svm, color='green', label='SVM(min,epoch) = ({}, {})'.format(round(svm_losses.min(),2), best_svm))

        # loss_img_array.plot(list_log_iteration, xgb_losses, label='XGB Acc')  # , cmap='plasma'  )

        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=7)
        plt.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()

        if save_fig:

            plt.savefig(
                figs_path
                + TODAY
                + "/"
                + cache_prefix
                + "_eval-Recall"
                + str(list_log_iteration[-1])
                + ".pdf",
                dpi=600,
            )

        if SHOW_PLOTS:
            plt.show()

        plt.close(f)
        # =======================================================================================================================================
        # =======================================================================================================================================
        # =======================================================================================================================================

        # f, loss_img_array = plt.subplots(1, 1, figsize=(3, 1.5))
        f, loss_img_array = plt.subplots(1, 1)

        loss_img_array.set_xlabel("Epochs")  # Add x label to all plots
        loss_img_array.set_ylabel("Loss")  # Add x label to all plots

        # print('Here Rizwann.....')

        # print(list_log_iteration)
        # print(epoch_list_gen_loss)

        loss_img_array.plot(
            list_log_iteration, epoch_list_gen_loss, label="L[G(z)]"
        )  # , cmap='plasma'  )
        loss_img_array.plot(
            list_log_iteration, epoch_list_disc_loss_real, label="L[D(x)]"
        )  # , cmap='plasma'  )
        loss_img_array.plot(
            list_log_iteration, epoch_list_disc_loss_generated, label="L[D(G(z))]"
        )  # , cmap='plasma'  )

        # plt.legend(loc='upper left', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=4)

        plt.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=3,
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()

        if save_fig:

            plt.savefig(
                figs_path
                + TODAY
                + "/"
                + cache_prefix
                + "_Losses-"
                + str(list_log_iteration[-1])
                + ".pdf",
                dpi=600,
            )

        if SHOW_PLOTS:
            plt.show()

        plt.close(f)


# =======================================================================================================================================


# f, acc_img_array = plt.subplots(1, 1)


# acc_img_array.set_xlabel('Epochs')  # Add x label to all plots
# acc_img_array.set_ylabel('Accuracy')  # Add x label to all plots

# # acc_img_array.plot(list_log_iteration, epoch_list_gen_acc, label = 'G_acc')  # , cmap='plasma'  )
# acc_img_array.plot(list_log_iteration, epoch_list_disc_acc_real, label = 'D(x)_acc')  # , cmap='plasma'  )
# acc_img_array.plot(list_log_iteration, epoch_list_disc_acc_generated, label = 'D(G(z))_acc')  # , cmap='plasma'  )


# # plt.legend(loc='upper left', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=4)


# plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,
#            borderaxespad=0, frameon=False)

# plt.tight_layout()

# if save_fig:

# 	plt.savefig(figs_path + TODAY + '/' + cache_prefix + '_Losses-' + str(list_log_iteration[-1]) + '.pdf', dpi=600)

# if SHOW_PLOTS:
# 	plt.show()


# plt.close(f)

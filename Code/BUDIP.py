########################################################################################################################
# Project: blind unmixing using DIP techniques.
# In this file, we implement the BUDIP using ADIP and EDIP from their corresponding Modules.
########################################################################################################################

import os.path
import random
import shutil
import sys
import time
from timeit import default_timer as timer

import numpy as np
import scipy

sys.path.append("..")
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from pysptools.abundance_maps.amaps import FCLS
from scipy.io import loadmat, savemat
from torch.utils.tensorboard import SummaryWriter

from ADIP import ADIP_loss_torch, ADIP_torch_I
from EDIP import EDIP_loss_torch, EDIP_torch_I
from Losses import RMSE_metric, angle_distance_metric
from Real_Data_preprocessing import load_JasperRidge_data
from utils import (
    Endmember_extract,
    Endmember_reorder2,
    create_project_log_path,
    np_to_torch,
    plot_abundance_map,
    plot_Endm,
    summary2readme,
    torch_to_np,
    auto_select_GPU,
)


class Dot_torch(nn.Module):
    def __init__(self):
        super(Dot_torch, self).__init__()

    def forward(self, inputs):
        """
        :param inputs: [E_input, A_input], where
                        shape(A_input)=(1, num_bands, num_endm),
                        shape(A_input)=(1, num_endm, img_row, img_col)
        :return:
        """
        E_input, A_input = inputs
        abu = A_input[0]
        output = torch.tensordot(E_input, abu, dims=1)
        return output


class BUDIP_I_torch(nn.Module):
    def __init__(self, img_row, img_col, NO_Bands, NO_Endms):
        super(BUDIP_I_torch, self).__init__()
        """
        In this function, we define the BUDIP network using torch.
        :param img_row: number of row of HSI image
        :param img_col:  number of col of HSI image
        :param NO_Bands: number of bands of HSI image
        :param NO_Endms: number of endmembers of HSI image
        """
        # define EDIP
        input_shape = (NO_Bands, NO_Endms)
        output_shape = input_shape
        num_filters_down = [256, 512, 1024]
        num_filters_up = [512, 256, input_shape[0]]
        filter_size_up = [3, 3, 3]
        filter_size_down = [3, 3, 3]
        filter_size_skip = 1
        downsample_modes = ["stride", "stride", "stride"]
        strides = [2, 1, 1]
        activation = "leaky_relu"
        padding = 1
        self.E_net = EDIP_torch_I(
            input_shape,
            output_shape,
            num_filters_up,
            num_filters_down,
            filter_size_up,
            filter_size_down,
            filter_size_skip,
            downsample_modes,
            strides,
            activation,
            padding,
        )

        # # define ADIP
        input_shape = (NO_Endms, img_row, img_col)
        output_shape = input_shape
        num_filters_down = [256, 512, 1024]
        num_filters_up = [512, 256, input_shape[0]]
        filter_size_up = [3, 3, 3]
        filter_size_down = [3, 3, 3]
        filter_size_skip = 1
        downsample_modes = ["stride", "stride", "stride"]
        strides = [2, 1, 1]
        activation = "leaky_relu"
        padding = 1
        self.A_net = ADIP_torch_I(
            input_shape,
            output_shape,
            num_filters_up,
            num_filters_down,
            filter_size_up,
            filter_size_down,
            filter_size_skip,
            downsample_modes,
            strides,
            activation,
            padding,
        )

        # ---------------linear mix ------------------
        self.Dot = Dot_torch()

    def forward(self, inputs):
        E_input, A_input = inputs

        A_output = self.A_net(A_input)
        E_output = self.E_net(E_input)

        output = self.Dot([E_output, A_output])

        return E_output, A_output, output


def main():
    Endm_ext_method = "SiVM"  # VCA, SiVM, uDAS, EGU_Net, GT, BUDDIP, EDAA, HyperCSI
    dataset_name = "JasperRidge"  # 'Synthetic', 'Purity_Synthetic', 'JasperRidge', 'Urban6', 'Samson'

    ##--------------------------------------------------------------------------------------
    # 1. read data
    if dataset_name == "JasperRidge":
        # a. JasperRidge
        hsi_data, true_endm_sig, true_abundances, data_params = load_JasperRidge_data()
        log_dir = f"../result/{Endm_ext_method}/"

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    hsi_data = hsi_data.astype(np.float32)
    true_endm_sig = true_endm_sig.astype(np.float32)
    true_abundances = true_abundances.astype(np.float32)

    NO_Bands, NO_Endms = true_endm_sig.shape
    img_row, img_col = data_params["img_size"]
    img_row, img_col = np.int32(img_row), np.int32(img_col)
    ##--------------------------------------------------------------------------------------
    ##################################
    # endm extraction:
    # 1. VCA
    # 2. SiVM from UnDIP
    ##################################
    if Endm_ext_method == "SiVM":
        # 2. SiVM (from UnDIP) to extract est_endm
        img_resh = hsi_data.T
        V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
        PC = np.diag(SS) @ U
        img_resh_DN = V[:, :NO_Endms] @ PC[:NO_Endms, :]
        img_resh_np_clip = np.clip(img_resh_DN, 0, 1)
        II, III = Endmember_extract(img_resh_np_clip, NO_Endms)
        E_np1 = img_resh_np_clip[:, II]
        asq = Endmember_reorder2(true_endm_sig, E_np1)
        est_endm = E_np1[:, asq]

    ##################################
    # abundance estimation:
    # FCLS
    ##################################
    fcls_abundance = FCLS(hsi_data, est_endm.T)
    # avoid overflow when calc angle loss
    fcls_abundance = np.clip(fcls_abundance, 1e-6, 1 - (1e-6))

    ##--------------------------------------------------------------------------------------
    # prepare inputs and outputs
    x_Abundance = np.reshape(fcls_abundance, (img_row, img_col, NO_Endms))
    x_Abundance = np.transpose(x_Abundance, axes=[2, 0, 1]).astype(np.float32)

    y_Abundance = np.reshape(true_abundances, (img_row, img_col, NO_Endms))
    y_Abundance = np.transpose(y_Abundance, axes=[2, 0, 1]).astype(np.float32)

    x_Endmember = est_endm.astype(np.float32)

    y_Endmember = true_endm_sig.astype(np.float32)

    y = np.reshape(hsi_data, (img_row, img_col, NO_Bands))
    y = np.transpose(y, (2, 0, 1)).astype(np.float32)

    EDIP_Input = x_Endmember
    ADIP_Input = x_Abundance

    data = [
        EDIP_Input,
        ADIP_Input,
        x_Abundance,
        y_Abundance,
        x_Endmember,
        y_Endmember,
        y,
    ]

    ##--------------------------------------------------------------------------------------
    def train_test_model(data, run_name, hparams, sema):
        seed = int(time.time())
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"\r\nseed = {seed}\r\n")
        # choose a GPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        (
            EDIP_Input,
            ADIP_Input,
            x_Abundance,
            y_Abundance,
            x_Endmember,
            y_Endmember,
            y,
        ) = data

        # move the data into gpu
        EDIP_Input_torch = np_to_torch(EDIP_Input).to(device)
        ADIP_Input_torch = np_to_torch(ADIP_Input).to(device)
        x_Abundance_torch = np_to_torch(x_Abundance).to(device)
        x_Endmember_torch = np_to_torch(x_Endmember).to(device)
        y_torch = np_to_torch(y).to(device)

        # build net

        net = BUDIP_I_torch(img_row, img_col, NO_Bands, NO_Endms).to(device)

        # define loss and opt
        EDIP_MSE_criterion = EDIP_loss_torch(x_Abundance_torch, y_torch)
        ADIP_MSE_criterion = ADIP_loss_torch(x_Endmember_torch, y_torch)
        BUDIP_MSE_criterion = nn.MSELoss()

        optimizer = optim.Adam(net.parameters(), lr=hparams["lr"])

        # ---------------------------- experiment log ----------------------------
        Readme = (
            "Try Blind unmixing on real Dataset.\r\n"
            + "Model invovled: reduced BUDIP.\r\n"
            + "data_params:\r\n"
            + str(data_params)
            + "\r\n"
            + run_name
            + "\r\n"
            + "seed: "
            + str(seed)
            + "\r\n"
            + "Endmember extraction method: "
            + Endm_ext_method
            + "\r\n"
            + str(hparams)
        )

        kwargs = {
            "Readme": Readme,
            "EDIP_mse_weight": hparams["EDIP_loss_weight"],
            "ADIP_mse_weight": hparams["ADIP_loss_weight"],
            "BUDIP_mse_weight": hparams["BUDIP_loss_weight"],
            "lr": hparams["lr"],
        }
        (
            program_log_path,
            model_checkpoint_dir,
            tensorboard_log_dir,
            model_log_dir,
        ) = create_project_log_path(project_path=log_dir + run_name, **kwargs)

        # define tensorboard writer
        writer = SummaryWriter(tensorboard_log_dir)

        # the SAD of est_endm and the rmse of est_abu
        sad_endmwise, sad_guidance = angle_distance_metric(
            y_Endmember.T, x_Endmember.T, verbose=True
        )
        summary_str = (
            Endm_ext_method
            + " est endmember SAD: %f\r\n" % (sad_guidance)
            + f"endmember-wise SAD : {sad_endmwise}"
        )
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        rmse_guidance = RMSE_metric(true_abundances, fcls_abundance)
        summary_str = "FCLS est abundance RMSE: %f" % (rmse_guidance)
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")
        aad_guidance = angle_distance_metric(true_abundances, fcls_abundance)
        summary_str = "FCLS est abundance AAD: %f" % (aad_guidance)
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        # plot the input
        plt.figure()
        plt.plot(EDIP_Input)
        plt.title("E_input: SAD=%f" % (sad_guidance))
        plt.savefig(program_log_path + "E_input.png")
        plt.close("all")

        plot_abundance_map(
            np.reshape(true_abundances, (img_row, img_col, NO_Endms)),
            np.reshape(fcls_abundance, (img_row, img_col, NO_Endms)),
            filepath=program_log_path + "A_Input ",
            suptitle="rmse=%f" % (rmse_guidance),
        )
        plt.close("all")

        # train
        num_epochs = hparams["num_epochs"]
        EDIP_mseLoss_Weight = hparams["EDIP_loss_weight"]
        ADIP_mseLoss_Weight = hparams["ADIP_loss_weight"]
        BUDIP_mseLoss_Weight = hparams["BUDIP_loss_weight"]
        best_RMSE, best_AAD, best_SAD = np.inf, np.inf, np.inf
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            E_output, A_output, output = net([EDIP_Input_torch, ADIP_Input_torch])

            E_mse_loss = EDIP_MSE_criterion(x_Endmember_torch, E_output)
            A_mse_loss = ADIP_MSE_criterion(x_Abundance_torch, A_output)
            BU_mse_loss = BUDIP_MSE_criterion(output, y_torch)

            loss = (
                EDIP_mseLoss_Weight * E_mse_loss
                + ADIP_mseLoss_Weight * A_mse_loss
                + BUDIP_mseLoss_Weight * BU_mse_loss
            )

            loss.backward()
            optimizer.step()

            # print training process
            if (epoch + 1) % 100 == 0 or epoch == 0:
                A_output_np = torch_to_np(A_output)
                E_output_np = torch_to_np(E_output)
                rmse = RMSE_metric(
                    y_Abundance.transpose((1, 2, 0)).reshape(
                        img_row * img_col, NO_Endms
                    ),
                    A_output_np.transpose((1, 2, 0)).reshape(
                        img_row * img_col, NO_Endms
                    ),
                )
                aad = angle_distance_metric(
                    y_Abundance.transpose((1, 2, 0)).reshape(
                        img_row * img_col, NO_Endms
                    ),
                    A_output_np.transpose((1, 2, 0)).reshape(
                        img_row * img_col, NO_Endms
                    ),
                )
                sad = angle_distance_metric(y_Endmember.T, E_output_np.T)

                best_AAD = np.minimum(best_AAD, aad)
                if best_RMSE > rmse:
                    best_RMSE = rmse
                    savemat(
                        tensorboard_log_dir + "best est_abundance.mat",
                        {"est_abundance": A_output_np},
                    )
                if best_SAD > sad:
                    best_SAD = sad
                    savemat(
                        tensorboard_log_dir + "best est_endmember.mat",
                        {"est_endmember": E_output_np},
                    )
                    torch.save(net.state_dict(), model_checkpoint_dir + "model.pth")

                summary_str = (
                    "epoch %d/%d: loss = %f, EDIP_loss = %f, ADIP_loss = %f, BUDIP_loss = %f, rmse = %f, aad = %f, sad = %f"
                    % (
                        epoch + 1,
                        num_epochs,
                        loss.item(),
                        E_mse_loss.item(),
                        A_mse_loss.item(),
                        BU_mse_loss.item(),
                        rmse,
                        aad,
                        sad,
                    )
                )
                print(summary_str)
                summary2readme(summary_str, program_log_path + "Readme.txt")

                # ...log the running loss into board
                writer.add_scalar("total mse loss", loss.item(), epoch + 1)
                writer.add_scalar("EDIP mse loss", E_mse_loss.item(), epoch + 1)
                writer.add_scalar("ADIP mse loss", A_mse_loss.item(), epoch + 1)
                writer.add_scalar("BUDIP mse loss", BU_mse_loss.item(), epoch + 1)

                writer.add_scalar("abundance RMSE", rmse, epoch + 1)
                writer.add_scalar("abundance AAD", aad, epoch + 1)
                writer.add_scalar("endmember SAD", sad, epoch + 1)

                # ...log the est endm and abudance
                fig = plot_Endm(
                    y_Endmember,
                    E_output_np,
                    suptitle="epoch %d: sad= %f" % (epoch + 1, sad),
                )
                writer.add_figure("est endm", fig, epoch + 1, close=True)
                plt.close("all")

                fig = plot_abundance_map(
                    y_Abundance.transpose((1, 2, 0)),
                    A_output_np.transpose((1, 2, 0)),
                    suptitle="epoch %d: rmse= %f" % (epoch + 1, rmse),
                )
                writer.add_figure("abu map", fig, epoch + 1, close=True)
                plt.close("all")

        summary_str = (
            f"best rmse = {best_RMSE:.4f}, aad = {best_AAD:.4f}, sad = {best_SAD:.4f}"
        )
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        best_endmember = loadmat(tensorboard_log_dir + "best est_endmember.mat")[
            "est_endmember"
        ]
        sad_endmwise, _ = angle_distance_metric(
            y_Endmember.T, best_endmember.T, verbose=True
        )
        summary_str = f"endmember wise SAD = {sad_endmwise}"
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        # calc test time
        start = timer()
        _, _, _ = net([EDIP_Input_torch, ADIP_Input_torch])
        test_time = timer() - start
        summary_str = f"test time: {test_time} s"
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        # close the writer after use
        writer.flush()
        writer.close()

        if best_AAD > aad_guidance and best_SAD > sad_guidance:
            print("delete failed training.")
            shutil.rmtree(program_log_path)

        # return train_test_success
        del net, writer
        sema.release()

    session_num = 0
    processes = []
    sema = mp.Semaphore(value=5)
    num_trials = 5

    for trial in range(num_trials):
        hparams = {
            "EDIP_loss_weight": 1.0,
            "ADIP_loss_weight": 1.0,
            "BUDIP_loss_weight": 1.0,
            "lr": 1e-3,
            "num_epochs": 24000,
        }

        sema.acquire()
        session_num += 1
        run_name = f"run-{session_num}-"
        print("--- Starting trial: %s" % run_name)
        print(hparams)
        p = mp.Process(target=train_test_model, args=(data, run_name, hparams, sema))
        p.start()
        processes.append(p)
        time.sleep(10)


if __name__ == "__main__":
    main()

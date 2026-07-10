import json
import os
import sys
import time
import sd
import argparse
import numpy as np
import uuid
import pprint
import random

sys.path.append("E:\\onsemi\\onsemi_thrid_part")

offset_table = {
    "E7111V2_2.1.50&E7111V2":{
        "hg":-30,
        "lg":-30,
        "ht":20,
        "lt":20,
        "et":20
    },
    "E7111V2_2.1.50&CharmV2-0512-G630":{
        "hg":-30,
        "lg":-30,
        "ht":20,
        "lt":20,
        "et":20
    },
    "E7111V2_2.2.64&Condor":{
        "hg":-30,
        "lg":-30,
        "ht":86,
        "lt":20,
        "et":20
    },
    "E7111V2_2.7.201&F5300 2.2.7":{
        "hg":-30,
        "lg":-30,
        "ht":20,
        "lt":20,
        "et":20
    },
    "E7160SL_1.18.1670&RIC Charm W 1181670":{
        "hg":-30,
        "lg":-30,
        "ht":20,
        "lt":20,
        "et":20
    },
    "E7160SL_1.18.1670&BTE Co 12W-118":{
        "hg":-30,
        "lg":-30,
        "ht":20,
        "lt":20,
        "et":20
    },
    "E7160SL_1.13.1507&Charm K830W":{
        "hg":-30,
        "lg":-20,
        "ht":20,
        "lt":40,
        "et":20
    },
}

lib_mapping = {
    "E7111V2_2.1.50":"..\\libs\\CharmV2-0512-G630.library",
    "E7111V2_2.2.64":"..\\libs\\Condor.library",
    "E7111V2_2.7.201":"..\\libs\\F5300 2.2.7.library",
    "E7160SL_1.13.1507":"..\\libs\\Charm K830W.library",
    "E7160SL_1.18.1670":"..\\libs\\BTE Co 12W-118.library",
    "E7160SL_1.18.1670_RIC":"..\\libs\\RIC Charm W 1181670.library"
}

pm = sd.ProductManager()

class HiproNoAids:
    def __init__(self, mem=0):
        self.__config()
        self.mem = mem
        self.product = None
        self.init_lib_params_file_path = "init_lib_params.json"
        with open(self.init_lib_params_file_path, "r") as f:
            self.initial_lib_params = json.load(f)

    def set_chip(self, chip):
        self.chip = chip

    def set_sdk(self, lib_path, lr):
        self._library_path = lib_path
        self.side = sd.kLeft if lr == 'l' else sd.kRight

    def initial(self, spec_chip):
        self.library = pm.LoadLibraryFromFile(self._library_path)
        try:
            if self.library.HasKey:
                self.library.UnlockLibrary(self.__key)
        except Exception as e:
            print("inner", e)
            # traceback.print_exc()
        self.product = self.library.Products.GetById(self.mem).CreateProduct()
        self.init_lib_params(spec_chip)


    def __config(self):
        self._current_dir = os.path.abspath(os.path.dirname(__file__))
        self._bin_dir = "../../extra_files/sd_dir"
        self.__key = "89703236"


    def write_parameters(self, mem, kvs):
        if mem == -3:
            prams = self.product.SystemMemory.Parameters
        else:
            prams = self.product.Memories[mem].Parameters
        for k, v in kvs.items():
            # logger.info(f"{k}: {v}")
            try:
                i = prams.GetById(k)
            except Exception as e:
                print(e)
                print("E_UNKNOWN_ID:", k)
                continue
            try:
                if i.Type in {1, 4, 5}:
                    i.Value = round(float(v))
                elif i.Type == 3:
                    i.BooleanValue = round(float(v))
                elif i.Type == 2:
                    i.DoubleValue = float(v)
                else:
                    raise ValueError("No such type support")
            except Exception as e:
                print(e)
                raise ValueError("parameter:{} = {} is wrong".format(k, v))
            
    def get_param_range(self, chip_valid_param_dict:dict)->dict:
        param_range = {"ParametersDouble0":{}, "ParametersLong0":{}, "SystemParametersDouble":{}, "SystemParametersLong":{}}
        for param_type, param_name_list in chip_valid_param_dict.items():
            if "SystemParameters" in param_type:
                prams = self.product.SystemMemory.Parameters
            else:
                prams = self.product.Memories[0].Parameters
            for k in param_name_list:
                try:
                    i = prams.GetById(k)
                except Exception as e:
                    print(e)
                    print("[get_param_range] E_UNKNOWN_ID:", k)
                    # continue
                try:
                    if i.Type in {1, 4, 5}:
                        param_range[param_type][k] = {"max":i.Max, "min":i.Min, "type":"int"}
                    elif i.Type == 3:
                        param_range[param_type][k] = {"max":i.Max, "min":i.Min, "type":"bool"}
                    elif i.Type == 2:
                        param_range[param_type][k] = {"max":i.DoubleMax, "min":i.DoubleMin, "type":"double"}
                    else:
                        raise ValueError("No such type support") 
                except Exception as e:
                    print(e)
                    # raise ValueError("parameter:{} = {} is wrong".format(k, v))
        return param_range

    def get_single_param_range(self, k:str, param_type:str):
        if param_type == "SystemParameters":
            prams = self.product.SystemMemory.Parameters
        elif param_type == "Parameters":
            prams = self.product.Memories[0].Parameters
        try:
            i = prams.GetById(k)
        except Exception as e:
            print(e)
            print("E_UNKNOWN_ID:", k)
            # continue
        try:
            if i.Type in {1, 4, 5}:
                return i.Value
            elif i.Type == 3:
                return i.Value
            elif i.Type == 2:
                return i.DoubleValue
            else:
                raise ValueError("No such type support") 
        except Exception as e:
            print(e)
    
    def init_lib_params(self, spec_chip = None):
        if spec_chip != None:
            initial_lib_params = self.initial_lib_params[spec_chip]
        else:
            initial_lib_params = self.initial_lib_params[self.chip]
        if self.chip == "E7160SL_1.18.1670":
            self.e7160sl_118_init_params = {
                "ALD617_BTE_Front@ALD617_BTE_SPK_37AP":initial_lib_params["condor"],
                "BTE P Bowen FM@BTE P Bowen":initial_lib_params["bowen"],
                "50PC33 MIC@31570":initial_lib_params["fascinating"],
                "50PC33 MIC@30008":initial_lib_params["skylark"],
            }
            return
        for param_type in initial_lib_params:
            if "SystemParameters" in param_type:
                self.write_parameters(-3, initial_lib_params[param_type])
            else:
                self.write_parameters(self.mem, initial_lib_params[param_type])

    def init_7160sl_118_params(self, mic_rec):
        init_param = self.e7160sl_118_init_params[mic_rec]
        for param_type in init_param:
            if "SystemParameters" in param_type:
                self.write_parameters(-3, init_param[param_type])
            else:
                self.write_parameters(self.mem, init_param[param_type])

    def get_curve(self, line_type=1):
        curves = {}
        try:
            gd = self.product.Graphs.GetById(line_type)
            gh = gd.CreateGraph()
            st = gh.GraphSettings
            # freqs = [125, 250, 350, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000]
            freqs = [200,210,223,236,250,265,281,297,315,334,354,375,397,420,445,472,500,530,561,595,630,667,707,749,794,841,891,944,1000,1059,1122,
                    1189,1260,1335,1414,1498,1587,1682,1782,1888,2000,2119,2245,2378,2520,2670,2828,2997,3175,3364,3564,3775,4000,4238,4490,4757,
                    5040,5339,5657,5993,6350,6727,7127,7551,8000]
            input_levels = {"50": 50.0, "80": 80., "90": 90.}
            for dv in input_levels:
                if st.ContainsId("InputLevel"):
                    il = st.GetById("InputLevel")
                    il.DoubleValue = input_levels[dv]
                gh.SetDomain(len(freqs), freqs)
                res = gh.CalculatePoints(len(freqs))
                curves[dv] = res
        except Exception as e:
            print("in get_curve:", e)
            # traceback.print_exc()
        return curves

    def set_mic_rec(self, mic, rec):
        model = self.library.TransducerModels
        res = model.GetByIdAndType(mic, sd.kMicrophoneSensitivityModel)
        self.product.SetTransducerModel(sd.kMicrophone1, res)
        res = model.GetByIdAndType(rec, sd.kReceiverSensitivityModel)
        self.product.SetTransducerModel(sd.kReceiver1, res)
        res = model.GetByIdAndType(rec, sd.kReceiverSaturationModel)
        self.product.SetTransducerModel(sd.kReceiver1, res)


def get_parser():
    parser = argparse.ArgumentParser(description='parse argument')
    parser.add_argument("-l", "--lib_path", default="../CharmV2-0512-G630.library")
    parser.add_argument("--lr", default="left")
    parser.add_argument("--chip", help="must specify chip name and version, such as E7111V2_2.1.50, E7160SL_1.181670 etc.")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    chip = args.chip
    lib_path = args.lib_path
    
    # ===== 给定一组参数，输出频响曲线 ======
    param_en_dict = {'X_HC_PostBiquad0Coeffa1': -0.3552, 'X_HC_PostBiquad0Coeffa2': 0.0719, 'X_HC_PostBiquad0Coeffb0': 0.3208, 'X_HC_PostBiquad0Coeffb1': 0.6417, 'X_HC_PostBiquad0Coeffb2': 0.3208, 'X_HC_PreBiquad0Coeffa1': 1.8401, 'X_HC_PreBiquad0Coeffa2': -0.849, 'X_HC_PreBiquad0Coeffb0': 0.9223, 'X_HC_PreBiquad0Coeffb1': -1.8445, 'X_HC_PreBiquad0Coeffb2': 0.9223, 'X_AuxiliaryAttenuation': 0, 'X_MicrophoneAttenuation': 4.478897608350962e-06, 'X_WDRC_ChannelOutputLimit[0]': 48, 'X_WDRC_ChannelOutputLimit[1]': 53, 'X_WDRC_ChannelOutputLimit[2]': 46, 'X_WDRC_ChannelOutputLimit[3]': 46, 'X_WDRC_ChannelOutputLimit[4]': 47, 'X_WDRC_ChannelOutputLimit[5]': 50, 'X_WDRC_ChannelOutputLimit[6]': 55, 'X_WDRC_ChannelOutputLimit[7]': 47, 'X_WDRC_ChannelOutputLimit[8]': 55, 'X_WDRC_ChannelOutputLimit[9]': 56, 'X_WDRC_ChannelOutputLimit[10]': 55, 'X_WDRC_ChannelOutputLimit[11]': 51, 'X_WDRC_ChannelOutputLimit[12]': 60, 'X_WDRC_ChannelOutputLimit[13]': 51, 'X_WDRC_ChannelOutputLimit[14]': 58, 'X_WDRC_ChannelOutputLimit[15]': 55, 'X_WDRC_ExpansionEnable[0]': 1, 'X_WDRC_ExpansionEnable[1]': 1, 'X_WDRC_ExpansionEnable[2]': 1, 'X_WDRC_ExpansionEnable[3]': 1, 'X_WDRC_ExpansionEnable[4]': 1, 'X_WDRC_ExpansionEnable[5]': 1, 'X_WDRC_ExpansionEnable[6]': 1, 'X_WDRC_ExpansionEnable[7]': 1, 'X_WDRC_ExpansionEnable[8]': 1, 'X_WDRC_ExpansionEnable[9]': 1, 'X_WDRC_ExpansionEnable[10]': 1, 'X_WDRC_ExpansionEnable[11]': 1, 'X_WDRC_ExpansionEnable[12]': 1, 'X_WDRC_ExpansionEnable[13]': 1, 'X_WDRC_ExpansionEnable[14]': 1, 'X_WDRC_ExpansionEnable[15]': 1, 'X_EQ_ChannelGain_dB[0]': 12, 'X_EQ_ChannelGain_dB[1]': 15, 'X_EQ_ChannelGain_dB[2]': 16, 'X_EQ_ChannelGain_dB[3]': 15, 'X_EQ_ChannelGain_dB[4]': 17, 'X_EQ_ChannelGain_dB[5]': 13, 'X_EQ_ChannelGain_dB[6]': 16, 'X_EQ_ChannelGain_dB[7]': 8, 'X_EQ_ChannelGain_dB[8]': 21, 'X_EQ_ChannelGain_dB[9]': 19, 'X_EQ_ChannelGain_dB[10]': 16, 'X_EQ_ChannelGain_dB[11]': 15, 'X_EQ_ChannelGain_dB[12]': 20, 'X_EQ_ChannelGain_dB[13]': 15, 'X_EQ_ChannelGain_dB[14]': 17, 'X_EQ_ChannelGain_dB[15]': 19, 'X_EQ_ChannelGain_dB[16]': 18, 'X_EQ_ChannelGain_dB[17]': 12, 'X_EQ_ChannelGain_dB[18]': 16, 'X_EQ_ChannelGain_dB[19]': 15, 'X_EQ_ChannelGain_dB[20]': 13, 'X_EQ_ChannelGain_dB[21]': 16, 'X_EQ_ChannelGain_dB[22]': 16, 'X_EQ_ChannelGain_dB[23]': 19, 'X_EQ_ChannelGain_dB[24]': 23, 'X_EQ_ChannelGain_dB[25]': 22, 'X_EQ_ChannelGain_dB[26]': 23, 'X_EQ_ChannelGain_dB[27]': 23, 'X_EQ_ChannelGain_dB[28]': 21, 'X_EQ_ChannelGain_dB[29]': 18, 'X_EQ_ChannelGain_dB[30]': 13, 'X_EQ_ChannelGain_dB[31]': 15, 'X_EQ_ChannelGain_dB[32]': 16, 'X_EQ_ChannelGain_dB[33]': 15, 'X_EQ_ChannelGain_dB[34]': 8, 'X_EQ_ChannelGain_dB[35]': 6, 'X_EQ_ChannelGain_dB[36]': 1, 'X_EQ_ChannelGain_dB[37]': 1, 'X_EQ_ChannelGain_dB[38]': 0, 'X_EQ_ChannelGain_dB[39]': 0, 'X_EQ_ChannelGain_dB[40]': 0, 'X_EQ_ChannelGain_dB[41]': 1, 'X_EQ_ChannelGain_dB[42]': 1, 'X_EQ_ChannelGain_dB[43]': 1, 'X_EQ_ChannelGain_dB[44]': 1, 'X_EQ_ChannelGain_dB[45]': 0, 'X_EQ_ChannelGain_dB[46]': 26, 'X_EQ_ChannelGain_dB[47]': 14, 'X_EQ_ChannelGain_dB[48]': 0, 'X_FBC_ActiveSensitivity': 4, 'X_FBC_ActiveSpeed': 1, 'X_FBC_ActiveTime': 14, 'X_FBC_Enable': 1, 'X_FBC_GainManagementEnable': 1, 'X_FBC_GainManagementLimit': 13, 'X_FBC_IdleSpeed': 6, 'X_FE_FEMode': 1, 'X_NR_MaxDepth[0]': 6, 'X_NR_MaxDepth[1]': 6, 'X_NR_MaxDepth[2]': 6, 'X_NR_MaxDepth[3]': 6, 'X_NR_MaxDepth[4]': 6, 'X_NR_MaxDepth[5]': 6, 'X_NR_MaxDepth[6]': 6, 'X_NR_MaxDepth[7]': 6, 'X_NR_MaxDepth[8]': 6, 'X_NR_MaxDepth[9]': 6, 'X_NR_MaxDepth[10]': 6, 'X_NR_MaxDepth[11]': 6, 'X_NR_MaxDepth[12]': 6, 'X_NR_MaxDepth[13]': 6, 'X_NR_MaxDepth[14]': 6, 'X_NR_MaxDepth[15]': 6, 'X_NR_MaxDepth[16]': 6, 'X_NR_MaxDepth[17]': 6, 'X_NR_MaxDepth[18]': 6, 'X_NR_MaxDepth[19]': 6, 'X_NR_MaxDepth[20]': 6, 'X_NR_MaxDepth[21]': 6, 'X_NR_MaxDepth[22]': 6, 'X_NR_MaxDepth[23]': 6, 'X_NR_MaxDepth[24]': 6, 'X_NR_MaxDepth[25]': 6, 'X_NR_MaxDepth[26]': 6, 'X_NR_MaxDepth[27]': 6, 'X_NR_MaxDepth[28]': 6, 'X_NR_MaxDepth[29]': 6, 'X_NR_MaxDepth[30]': 6, 'X_NR_MaxDepth[31]': 6, 'X_NR_MaxDepth[32]': 6, 'X_NR_MaxDepth[33]': 6, 'X_NR_MaxDepth[34]': 6, 'X_NR_MaxDepth[35]': 6, 'X_NR_MaxDepth[36]': 6, 'X_NR_MaxDepth[37]': 6, 'X_NR_MaxDepth[38]': 6, 'X_NR_MaxDepth[39]': 6, 'X_NR_MaxDepth[40]': 6, 'X_NR_MaxDepth[41]': 6, 'X_NR_MaxDepth[42]': 6, 'X_NR_MaxDepth[43]': 6, 'X_NR_MaxDepth[44]': 6, 'X_NR_MaxDepth[45]': 6, 'X_NR_MaxDepth[46]': 6, 'X_NR_MaxDepth[47]': 6, 'X_NR_MaxDepth[48]': 6, 'X_SG_Bandwidth': 0, 'X_SG_Centerband': 3, 'X_SG_EnableMode': 0, 'X_SG_Level': 88, 'X_AuxiliaryInput': 0, 'X_LIM_TargetGain_dB': 7, 'X_WDRC_HighLevelGain[0]': 13, 'X_WDRC_LowLevelGain[0]': 28, 'X_WDRC_HighLevelThreshold[0]': 76, 'X_WDRC_LowLevelThreshold[0]': 29, 'X_WDRC_ExpansionThreshold[0]': 10, 'X_WDRC_HighLevelGain[1]': 17, 'X_WDRC_LowLevelGain[1]': 31, 'X_WDRC_HighLevelThreshold[1]': 68, 'X_WDRC_LowLevelThreshold[1]': 27, 'X_WDRC_ExpansionThreshold[1]': 10, 'X_WDRC_HighLevelGain[2]': 23, 'X_WDRC_LowLevelGain[2]': 39, 'X_WDRC_HighLevelThreshold[2]': 72, 'X_WDRC_LowLevelThreshold[2]': 31, 'X_WDRC_ExpansionThreshold[2]': 10, 'X_WDRC_HighLevelGain[3]': 21, 'X_WDRC_LowLevelGain[3]': 37, 'X_WDRC_HighLevelThreshold[3]': 72, 'X_WDRC_LowLevelThreshold[3]': 35, 'X_WDRC_ExpansionThreshold[3]': 10, 'X_WDRC_HighLevelGain[4]': 14, 'X_WDRC_LowLevelGain[4]': 32, 'X_WDRC_HighLevelThreshold[4]': 74, 'X_WDRC_LowLevelThreshold[4]': 31, 'X_WDRC_ExpansionThreshold[4]': 10, 'X_WDRC_HighLevelGain[5]': 30, 'X_WDRC_LowLevelGain[5]': 46, 'X_WDRC_HighLevelThreshold[5]': 72, 'X_WDRC_LowLevelThreshold[5]': 32, 'X_WDRC_ExpansionThreshold[5]': 10, 'X_WDRC_HighLevelGain[6]': 22, 'X_WDRC_LowLevelGain[6]': 38, 'X_WDRC_HighLevelThreshold[6]': 69, 'X_WDRC_LowLevelThreshold[6]': 32, 'X_WDRC_ExpansionThreshold[6]': 10, 'X_WDRC_HighLevelGain[7]': 14, 'X_WDRC_LowLevelGain[7]': 30, 'X_WDRC_HighLevelThreshold[7]': 63, 'X_WDRC_LowLevelThreshold[7]': 24, 'X_WDRC_ExpansionThreshold[7]': 10, 'X_WDRC_HighLevelGain[8]': 18, 'X_WDRC_LowLevelGain[8]': 35, 'X_WDRC_HighLevelThreshold[8]': 77, 'X_WDRC_LowLevelThreshold[8]': 32, 'X_WDRC_ExpansionThreshold[8]': 10, 'X_WDRC_HighLevelGain[9]': 22, 'X_WDRC_LowLevelGain[9]': 39, 'X_WDRC_HighLevelThreshold[9]': 75, 'X_WDRC_LowLevelThreshold[9]': 29, 'X_WDRC_ExpansionThreshold[9]': 10, 'X_WDRC_HighLevelGain[10]': 21, 'X_WDRC_LowLevelGain[10]': 36, 'X_WDRC_HighLevelThreshold[10]': 72, 'X_WDRC_LowLevelThreshold[10]': 31, 'X_WDRC_ExpansionThreshold[10]': 10, 'X_WDRC_HighLevelGain[11]': 20, 'X_WDRC_LowLevelGain[11]': 38, 'X_WDRC_HighLevelThreshold[11]': 80, 'X_WDRC_LowLevelThreshold[11]': 34, 'X_WDRC_ExpansionThreshold[11]': 10, 'X_WDRC_HighLevelGain[12]': 21, 'X_WDRC_LowLevelGain[12]': 38, 'X_WDRC_HighLevelThreshold[12]': 74, 'X_WDRC_LowLevelThreshold[12]': 22, 'X_WDRC_ExpansionThreshold[12]': 10, 'X_WDRC_HighLevelGain[13]': 15, 'X_WDRC_LowLevelGain[13]': 34, 'X_WDRC_HighLevelThreshold[13]': 80, 'X_WDRC_LowLevelThreshold[13]': 31, 'X_WDRC_ExpansionThreshold[13]': 10, 'X_WDRC_HighLevelGain[14]': 19, 'X_WDRC_LowLevelGain[14]': 32, 'X_WDRC_HighLevelThreshold[14]': 68, 'X_WDRC_LowLevelThreshold[14]': 30, 'X_WDRC_ExpansionThreshold[14]': 10, 'X_WDRC_HighLevelGain[15]': 26, 'X_WDRC_LowLevelGain[15]': 44, 'X_WDRC_HighLevelThreshold[15]': 59, 'X_WDRC_LowLevelThreshold[15]': 38, 'X_WDRC_ExpansionThreshold[15]': 10}
    hipro = HiproNoAids()
    spec_chip = None
    lib_path = lib_mapping[chip]
    mic_rec_str = "9446M@31570"
    if chip == "E7111V2_2.1.50" and mic_rec_str == "9446M@27926":
        lib_path = "..\\libs\\CharmV2-0512-B230.library"
        spec_chip = "E7111V2_2.1.50_HS_B230/E330"
    hipro.set_chip(chip)
    hipro.set_sdk(lib_path, "l")
    hipro.initial(spec_chip = None)
    hipro.set_mic_rec(mic_rec_str.split("@")[0], mic_rec_str.split("@")[1])
    if chip == "E7160SL_1.18.1670":
        hipro.init_7160sl_118_params(mic_rec_str)
    hipro.write_parameters(0, param_en_dict)
    curve = hipro.get_curve()
    all_point = []
    for k,v in curve.items():
        all_point+=list(v)
    print(all_point)
    # ============================================

    # ====== 读npz文件，输出curve，是否跟label相等 ========
    # all_chip_mic_rec = json.load(open("../mic_rec_sample.json", "r"))
    # chip_mic_rec = all_chip_mic_rec[chip]

    # sdk_param_name = ["X_HC_PostBiquad0Coeffa1", "X_HC_PostBiquad0Coeffa2", "X_HC_PostBiquad0Coeffb0", "X_HC_PostBiquad0Coeffb1", "X_HC_PostBiquad0Coeffb2", "X_HC_PreBiquad0Coeffa1", "X_HC_PreBiquad0Coeffa2", "X_HC_PreBiquad0Coeffb0", "X_HC_PreBiquad0Coeffb1", "X_HC_PreBiquad0Coeffb2", "X_AuxiliaryAttenuation", "X_MicrophoneAttenuation", "X_WDRC_ChannelOutputLimit[0]", "X_WDRC_ChannelOutputLimit[1]", "X_WDRC_ChannelOutputLimit[2]", "X_WDRC_ChannelOutputLimit[3]", "X_WDRC_ChannelOutputLimit[4]", "X_WDRC_ChannelOutputLimit[5]", "X_WDRC_ChannelOutputLimit[6]", "X_WDRC_ChannelOutputLimit[7]", "X_WDRC_ChannelOutputLimit[8]", "X_WDRC_ChannelOutputLimit[9]", "X_WDRC_ChannelOutputLimit[10]", "X_WDRC_ChannelOutputLimit[11]", "X_WDRC_ChannelOutputLimit[12]", "X_WDRC_ChannelOutputLimit[13]", "X_WDRC_ChannelOutputLimit[14]", "X_WDRC_ChannelOutputLimit[15]", "X_WDRC_ExpansionEnable[0]", "X_WDRC_ExpansionEnable[1]", "X_WDRC_ExpansionEnable[2]", "X_WDRC_ExpansionEnable[3]", "X_WDRC_ExpansionEnable[4]", "X_WDRC_ExpansionEnable[5]", "X_WDRC_ExpansionEnable[6]", "X_WDRC_ExpansionEnable[7]", "X_WDRC_ExpansionEnable[8]", "X_WDRC_ExpansionEnable[9]", "X_WDRC_ExpansionEnable[10]", "X_WDRC_ExpansionEnable[11]", "X_WDRC_ExpansionEnable[12]", "X_WDRC_ExpansionEnable[13]", "X_WDRC_ExpansionEnable[14]", "X_WDRC_ExpansionEnable[15]", "X_EQ_ChannelGain_dB[0]", "X_EQ_ChannelGain_dB[1]", "X_EQ_ChannelGain_dB[2]", "X_EQ_ChannelGain_dB[3]", "X_EQ_ChannelGain_dB[4]", "X_EQ_ChannelGain_dB[5]", "X_EQ_ChannelGain_dB[6]", "X_EQ_ChannelGain_dB[7]", "X_EQ_ChannelGain_dB[8]", "X_EQ_ChannelGain_dB[9]", "X_EQ_ChannelGain_dB[10]", "X_EQ_ChannelGain_dB[11]", "X_EQ_ChannelGain_dB[12]", "X_EQ_ChannelGain_dB[13]", "X_EQ_ChannelGain_dB[14]", "X_EQ_ChannelGain_dB[15]", "X_EQ_ChannelGain_dB[16]", "X_EQ_ChannelGain_dB[17]", "X_EQ_ChannelGain_dB[18]", "X_EQ_ChannelGain_dB[19]", "X_EQ_ChannelGain_dB[20]", "X_EQ_ChannelGain_dB[21]", "X_EQ_ChannelGain_dB[22]", "X_EQ_ChannelGain_dB[23]", "X_EQ_ChannelGain_dB[24]", "X_EQ_ChannelGain_dB[25]", "X_EQ_ChannelGain_dB[26]", "X_EQ_ChannelGain_dB[27]", "X_EQ_ChannelGain_dB[28]", "X_EQ_ChannelGain_dB[29]", "X_EQ_ChannelGain_dB[30]", "X_EQ_ChannelGain_dB[31]", "X_EQ_ChannelGain_dB[32]", "X_EQ_ChannelGain_dB[33]", "X_EQ_ChannelGain_dB[34]", "X_EQ_ChannelGain_dB[35]", "X_EQ_ChannelGain_dB[36]", "X_EQ_ChannelGain_dB[37]", "X_EQ_ChannelGain_dB[38]", "X_EQ_ChannelGain_dB[39]", "X_EQ_ChannelGain_dB[40]", "X_EQ_ChannelGain_dB[41]", "X_EQ_ChannelGain_dB[42]", "X_EQ_ChannelGain_dB[43]", "X_EQ_ChannelGain_dB[44]", "X_EQ_ChannelGain_dB[45]", "X_EQ_ChannelGain_dB[46]", "X_EQ_ChannelGain_dB[47]", "X_EQ_ChannelGain_dB[48]", "X_FBC_ActiveSensitivity", "X_FBC_ActiveSpeed", "X_FBC_ActiveTime", "X_FBC_Enable", "X_FBC_GainManagementEnable", "X_FBC_GainManagementLimit", "X_FBC_IdleSpeed", "X_FE_FEMode", "X_NR_MaxDepth[0]", "X_NR_MaxDepth[1]", "X_NR_MaxDepth[2]", "X_NR_MaxDepth[3]", "X_NR_MaxDepth[4]", "X_NR_MaxDepth[5]", "X_NR_MaxDepth[6]", "X_NR_MaxDepth[7]", "X_NR_MaxDepth[8]", "X_NR_MaxDepth[9]", "X_NR_MaxDepth[10]", "X_NR_MaxDepth[11]", "X_NR_MaxDepth[12]", "X_NR_MaxDepth[13]", "X_NR_MaxDepth[14]", "X_NR_MaxDepth[15]", "X_NR_MaxDepth[16]", "X_NR_MaxDepth[17]", "X_NR_MaxDepth[18]", "X_NR_MaxDepth[19]", "X_NR_MaxDepth[20]", "X_NR_MaxDepth[21]", "X_NR_MaxDepth[22]", "X_NR_MaxDepth[23]", "X_NR_MaxDepth[24]", "X_NR_MaxDepth[25]", "X_NR_MaxDepth[26]", "X_NR_MaxDepth[27]", "X_NR_MaxDepth[28]", "X_NR_MaxDepth[29]", "X_NR_MaxDepth[30]", "X_NR_MaxDepth[31]", "X_NR_MaxDepth[32]", "X_NR_MaxDepth[33]", "X_NR_MaxDepth[34]", "X_NR_MaxDepth[35]", "X_NR_MaxDepth[36]", "X_NR_MaxDepth[37]", "X_NR_MaxDepth[38]", "X_NR_MaxDepth[39]", "X_NR_MaxDepth[40]", "X_NR_MaxDepth[41]", "X_NR_MaxDepth[42]", "X_NR_MaxDepth[43]", "X_NR_MaxDepth[44]", "X_NR_MaxDepth[45]", "X_NR_MaxDepth[46]", "X_NR_MaxDepth[47]", "X_NR_MaxDepth[48]", "X_SG_Bandwidth", "X_SG_Centerband", "X_SG_EnableMode", "X_SG_Level", "X_AuxiliaryInput", "X_LIM_TargetGain_dB", "X_WDRC_HighLevelGain[0]", "X_WDRC_LowLevelGain[0]", "X_WDRC_HighLevelThreshold[0]", "X_WDRC_LowLevelThreshold[0]", "X_WDRC_ExpansionThreshold[0]", "X_WDRC_HighLevelGain[1]", "X_WDRC_LowLevelGain[1]", "X_WDRC_HighLevelThreshold[1]", "X_WDRC_LowLevelThreshold[1]", "X_WDRC_ExpansionThreshold[1]", "X_WDRC_HighLevelGain[2]", "X_WDRC_LowLevelGain[2]", "X_WDRC_HighLevelThreshold[2]", "X_WDRC_LowLevelThreshold[2]", "X_WDRC_ExpansionThreshold[2]", "X_WDRC_HighLevelGain[3]", "X_WDRC_LowLevelGain[3]", "X_WDRC_HighLevelThreshold[3]", "X_WDRC_LowLevelThreshold[3]", "X_WDRC_ExpansionThreshold[3]", "X_WDRC_HighLevelGain[4]", "X_WDRC_LowLevelGain[4]", "X_WDRC_HighLevelThreshold[4]", "X_WDRC_LowLevelThreshold[4]", "X_WDRC_ExpansionThreshold[4]", "X_WDRC_HighLevelGain[5]", "X_WDRC_LowLevelGain[5]", "X_WDRC_HighLevelThreshold[5]", "X_WDRC_LowLevelThreshold[5]", "X_WDRC_ExpansionThreshold[5]", "X_WDRC_HighLevelGain[6]", "X_WDRC_LowLevelGain[6]", "X_WDRC_HighLevelThreshold[6]", "X_WDRC_LowLevelThreshold[6]", "X_WDRC_ExpansionThreshold[6]", "X_WDRC_HighLevelGain[7]", "X_WDRC_LowLevelGain[7]", "X_WDRC_HighLevelThreshold[7]", "X_WDRC_LowLevelThreshold[7]", "X_WDRC_ExpansionThreshold[7]", "X_WDRC_HighLevelGain[8]", "X_WDRC_LowLevelGain[8]", "X_WDRC_HighLevelThreshold[8]", "X_WDRC_LowLevelThreshold[8]", "X_WDRC_ExpansionThreshold[8]", "X_WDRC_HighLevelGain[9]", "X_WDRC_LowLevelGain[9]", "X_WDRC_HighLevelThreshold[9]", "X_WDRC_LowLevelThreshold[9]", "X_WDRC_ExpansionThreshold[9]", "X_WDRC_HighLevelGain[10]", "X_WDRC_LowLevelGain[10]", "X_WDRC_HighLevelThreshold[10]", "X_WDRC_LowLevelThreshold[10]", "X_WDRC_ExpansionThreshold[10]", "X_WDRC_HighLevelGain[11]", "X_WDRC_LowLevelGain[11]", "X_WDRC_HighLevelThreshold[11]", "X_WDRC_LowLevelThreshold[11]", "X_WDRC_ExpansionThreshold[11]", "X_WDRC_HighLevelGain[12]", "X_WDRC_LowLevelGain[12]", "X_WDRC_HighLevelThreshold[12]", "X_WDRC_LowLevelThreshold[12]", "X_WDRC_ExpansionThreshold[12]", "X_WDRC_HighLevelGain[13]", "X_WDRC_LowLevelGain[13]", "X_WDRC_HighLevelThreshold[13]", "X_WDRC_LowLevelThreshold[13]", "X_WDRC_ExpansionThreshold[13]", "X_WDRC_HighLevelGain[14]", "X_WDRC_LowLevelGain[14]", "X_WDRC_HighLevelThreshold[14]", "X_WDRC_LowLevelThreshold[14]", "X_WDRC_ExpansionThreshold[14]", "X_WDRC_HighLevelGain[15]", "X_WDRC_LowLevelGain[15]", "X_WDRC_HighLevelThreshold[15]", "X_WDRC_LowLevelThreshold[15]", "X_WDRC_ExpansionThreshold[15]"]
    # npz_data_dir = "E:\\onsemi\\data\\E7160SL_1.18.1670_1\\0a2f6e52-3161-4ae8-a48f-b805737b3f02.npz"
    # data = np.load(npz_data_dir)
    # x = data["x"].tolist()
    # label = data["label"]
    # mic_rec = data["mic_rec"].tolist()

    # hipro = HiproNoAids(chip)
    # hipro.set_sdk(lib_path, "l")
    # hipro.initial()
    # for i in range(1000):
    #     mic_rec_str = chip_mic_rec[mic_rec[i]]
    #     if chip == "E7160SL_1.18.1670":
    #         hipro.init_7160sl_118_params(mic_rec_str)
    #     hipro.set_mic_rec(mic_rec_str.split("@")[0], mic_rec_str.split("@")[1])
    #     sdk_param = x[i]
    #     sdk_param_dict = {name : sdk_param[idx] for idx, name in enumerate(sdk_param_name)}
    #     hipro.write_parameters(0, sdk_param_dict)
    #     lb = label[i]
    #     curve_dict = hipro.get_curve()
    #     curve = []
    #     for k,v in curve_dict.items():
    #         curve+=list(v)
    #     curve = np.array(curve)
    #     mae = np.sum(np.abs(lb-curve))
    #     if mae > 0.1:
    #         print(f"[ERROR!] {i}")
    # ==========================================================
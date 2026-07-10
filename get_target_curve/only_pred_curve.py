import copy
import pickle
import json
import os
import random

import torch
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
# import sys
# sys.ath.append()
from yp_config import DEVICE, chip_rg, coeff_post_values, coeff_pre_values, chip_rg,chip_params_num
from nalNet import nal_feat
from ypnet_20260625 import YPNet

class RegressinHandler(BaseHandler):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model.to(DEVICE)
        self.chip = None
        self.tp = 126
        # self.mean_gap = 0
        self.out_dim = 369
        with open("./bag_251011.json", "r") as f:
            self.bag = json.load(f)
        with open("./nal_bag_251011.json", "r") as f:
            self.nal_bag = json.load(f)
        with open("./mic_rec_full_table.json", "r") as f:
            self.mic_rec_table = json.load(f)
        with open("./chip_ai_new_params.json", "r") as f:
            self.chip_ai_params = json.load(f)

            # chip_rg = {"e7111v2": 0, "audion6": 1, "audion16plus": 2, "ethos": 3, "e7111": 4, "e7160sl": 5, "audion16":6, "b300":7}
            # i = ["ethos", "e7111_1.2.839", "e7111v2_2.1.50", "e7111v2_2.2.64", "e7111v2_2.7.201", "e7160sl_1.11.1428", "e7160sl_1.13.1507", 
            #       "e7160sl_1.17.1635", "e7160sl_1.18.1670", "audion6", "audion16plus", "audion16", "b300"]
            self.chip_ai_params = {chip_rg[i.lower().split("_")[0]]:j for i,j in self.chip_ai_params.items()}

    def get_mic_rec(self, cd, pdt_name, out_look,srn,trumpetType):
        pdt_name = pdt_name.replace(out_look, "").strip().replace("-R", "R").replace("项目", "project")
        out_look = out_look.replace(" ", "")
        if srn=="simulate":
            srn1 = 0
        else:
            if len(srn)==9:
                srn1 = 0
            else:
                srn1 = srn[0]
        mic_rec = {"mic":"null", "rec":"null"}
        if out_look in self.mic_rec_table[cd]:
            pdt_name = pdt_name.replace(out_look, "").strip().upper()
            if pdt_name in self.mic_rec_table[cd][out_look]:
                if "no_srn" in self.mic_rec_table[cd][out_look][pdt_name]:
                    if cd == "e7111v2" and out_look == "RIC":
                        mic_rec = self.mic_rec_table[cd][out_look][pdt_name]["no_srn"][trumpetType]
                    else:
                        mic_rec = self.mic_rec_table[cd][out_look][pdt_name]["no_srn"]
                else:
                    srn1 = str(srn1)
                    if srn1 in self.mic_rec_table[cd][out_look][pdt_name]:
                        mic_rec = self.mic_rec_table[cd][out_look][pdt_name][srn1]
                    elif "default" in self.mic_rec_table[cd][out_look][pdt_name]:
                        mic_rec = self.mic_rec_table[cd][out_look][pdt_name]["default"]
                    else:
                        mic_rec = self.mic_rec_table[cd][out_look][pdt_name]
        mic = self.bag["bag_mic"].get("id_"+mic_rec["mic"], self.bag["bag_mic"].get("id_"+"null"))
        rec = self.bag["bag_rec"].get("id_" + mic_rec["rec"], self.bag["bag_mic"].get("id_" + "null"))
        return mic, rec


    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        tlts = []
        info_cats = []
        info_nums = []
        all_nals = []
        for req in requests:
            data = req.get("data")
            if data is None:
                data = req.get("body")
            print("input:", data)
            data = json.loads(data)
            age = data["age"]
            db = data["db"]
            phis = data["phis"]
            gender = data["gender"]
            outlook = data["outlook"]
            trumpetType = data["trumpetType"]
            wear_time = data.get("weartime", 0.)
            tlt_ = np.array(data["tlt"])
            tlt = np.array(tlt_).T
            gap = np.abs(tlt[:, 0] - tlt[:, 1])
            self.mean_gap = np.mean(gap)
            tp = "neuron"
            if np.sum(tlt[:, 1]) > 0:
                if np.mean(gap[1:4]) > 20:
                    if np.mean(tlt[1:4, 1] < 30):
                        tp = "cond"
                    else:
                        tp = "mix"
            else:
                # gap = np.ceil(4 + 5 * np.random.randn(10))
                # gap = np.ceil(4 + 5 * np.random.randn(10))
                gap = np.zeros(10)+5
                self.mean_gap = 0
            tlt = np.array([tlt_[0], gap.tolist()])
            srn = data["srn"]
            series = data["series"]
            chip = data["chip"].lower()
            gender = "null" if int(gender)>1 else gender
            db_ = self.bag["bag_db"].get("id_"+str(db))
            db_nal = self.nal_bag["bag_db"].get("id_"+str(db))
            gender_ = self.bag["bag_gender"].get("id_"+str(gender))
            gender_nal = self.nal_bag["bag_gender"].get("id_"+str(gender))
            phis_ = self.bag["bag_phis"].get("id_"+str(phis))
            phis_nal = self.nal_bag["bag_phis"].get("id_"+str(phis))
            outlook_ = self.bag["bag_outlook"].get("id_"+str(outlook))
            outlook_nal = self.nal_bag["bag_outlook"].get("id_"+str(outlook))
            self.tp = self.bag["bag_loss_type"].get("id_"+str(tp))
            # age_ = self.bag["bag_age"].get("id_"+str(age))
            mic, rec = self.get_mic_rec(chip,series,outlook,srn,trumpetType)
            phis_ = 4 if phis_>3 else 3
            info = [age, wear_time, gender_, phis_, db_, outlook_, mic, rec, self.tp]
            info_cat = [int(i) for i in info[2:]]
            info_num = np.array(info[:2])
            for i in info_cat:
                if int(i)<0 or int(i)>266:
                    return None,None,None
            chip = np.array([chip_rg[c] for c in [chip,]], dtype=np.int32)
            self.chip = torch.LongTensor(chip)
            # ["gender", "phis", "outlook", "db"]
            nals = [int(i) for i in [gender_nal, phis_nal, outlook_nal, db_nal]]
            tlts.append(tlt)
            info_cats.append(info_cat)
            info_nums.append(info_num)
            all_nals.append(nals)
        tlts = torch.FloatTensor(tlts).permute(0, 2, 1).to(DEVICE)
        tlts[tlts>0]+= 20
        tlts[tlts > 0] /= 70
        info_cats = torch.LongTensor(info_cats).to(DEVICE)
        info_nums = torch.FloatTensor(info_nums).to(DEVICE)
        all_nals = torch.LongTensor(all_nals).to(DEVICE)
        all_nals = (nal_feat(tlts, all_nals, tlts.shape[0]) + 20) / 70
        return tlts, info_nums.unsqueeze(0), info_cats.unsqueeze(0), all_nals

    def inference(self, data, *args, **kwargs):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        # print(self.chip)
        tlt, info_nums, info_cats, nals = data
        if info_nums is None or info_cats is None:
            return None
        outs = self.model.forward(tlt, info_nums, info_cats, nals, self.chip)
        return outs

    def fix_rate(self,param):
        param = np.array(param)
        if self.tp > 126:
            dis = 8
            if self.tp == 128:
                dis = 12
            dis = max(4, dis - round((self.mean_gap-20)/10))
            print("dis_value:", dis)
            if self.chip == 0 or self.chip == 5:
                gap = (param[16:32] - param[:16])-dis
                param[16:32] -= gap*0.2
                param[:16] += gap*0.8
            elif self.chip == 4:
                gap = (param[8:16] - param[:8]) - dis
                param[8:16] -= gap * 0.2
                param[:8] += gap * 0.8
            elif self.chip == 1:
                param[12:12+6] = 2+min(3, round(dis/3))
            elif self.chip == 2:
                param[16:16+16] = 2+min(3, round(dis/3))
            elif self.chip == 3:
                param[12:12+8] = 2+min(3, round(dis/3))
        return param.tolist()

    '''hlg, llg, lth必须为偶数, mth = (hth+lth)/2, mlg=(hlg+llg)/2, hth固定为96'''
    def process_b300(self, param):
        hlg_16 = param[51:51+16]
        hth_16 = param[67:67+16]
        llg_16 = param[83:83+16]
        lth_16 = param[99:99+16]
        mlg_16 = param[115:115+16]
        mth_16 = param[131:131+16]
        for i in range(16):
            hlg = hlg_16[i]
            hth = 96
            llg = llg_16[i]
            lth = lth_16[i]
            hlg = hlg if hlg % 2 == 0 else hlg + 1
            llg = llg if llg % 2 == 0 else llg + 1
            lth = lth if lth % 2 == 0 else lth + 1
            mth = (hth + lth) // 2
            mlg = (hlg + llg) // 2
            hlg_16[i] = hlg
            hth_16[i] = hth
            llg_16[i] = llg
            lth_16[i] = lth
            mlg_16[i] = mlg
            mth_16[i] = mth
        return param[:51] + hlg_16 + hth_16 + llg_16 + lth_16 + mlg_16 + mth_16
    

    def postprocess(self, preds):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        if preds is None:
            raise ValueError("parameters error!!!")
        all_preds = preds.cpu().detach().numpy().tolist()
        # print("len of preds:", len(all_preds), "all_preds:", all_preds)
        all_res = []
        chip_id_ = None
        for o_idx, params in enumerate(all_preds):
            chip_id_ = self.chip[o_idx].item()  # len(self.chip) = 1，因为推理阶段bs就是1，里面的值就是对应的chip id
            all_keys = self.chip_ai_params[chip_id_]["en"]
            all_keys_nu = self.chip_ai_params[chip_id_]["nu"]

            if chip_id_ in {0, 4, 5}:
                idx = 0
                res_v = []
                for key in all_keys:
                    if "X_HC" in key:
                        if "X_HC_PostBiquad0Coeffb0" in key:
                            value = np.argmax(params[idx:idx + 9])
                        elif "X_HC_PreBiquad0Coeffa1" in key:
                            value = np.argmax(params[idx:idx + 9])
                        else:
                            continue
                        norm_value = coeff_post_values[value] if key[-2:] == "b0" else coeff_pre_values[value]
                        norm_value = list(norm_value) + [0., 0., 1., 0., 0.] * 3
                        res_v += norm_value
                        idx += 9
                        continue
                    # norm_value = None
                    if "ParametersLong" in key:
                        v = round(params[idx])
                    else:
                        v = float(params[idx])
                    res_v.append(v)
                    idx += 1
                local_res = res_v[:chip_params_num[self.chip[o_idx].item()]]
            else:
                local_res = [int(i) for i in params[:chip_params_num[self.chip[o_idx].item()]]]

            if chip_id_ == 7:
                local_res = self.process_b300(local_res)
            
            all_res.append(local_res)

        out_res = []#list 里面是dict
        for frame in all_res:
            out_dict = {}
            for idx, key in enumerate(all_keys_nu):
                out_dict[key] = frame[idx]
            if chip_id_ == 7:  # 2026/05/12更新：对于B300(也叫52900)芯片，需要删除一些参数
                del out_dict[635] # 删除EXP_Enabled
                del out_dict[637] # 删除ANR_Enabled
                del out_dict[638] # 删除FBC_CH0_Enabled
            out_res.append(out_dict)
        # print("output params len:", len(out_res[0]))


        return out_res, all_res[0], all_preds


def _run_suffix():
    from datetime import datetime
    import os as _os
    today = datetime.now().strftime("%m%d")
    n = 1
    out = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "outputs")
    while _os.path.exists(_os.path.join(out, f"target_results_{today}_{n}.json")):
        n += 1
    return f"{today}_{n}"

if __name__ == "__main__":
    # data = {'age': 50, 'chip': 'B300', 'chip_version': '', 'company': 'austar', 'db': 1, 'gender': 0, 'lr': 0, 'mac': 'D0:65:78:12:08:3B', 'outlook': 'HS', 'phis': 0, 'series': 'Laverock 8', 'srn': '126171539', 'tlt': [[55, 60, 60, 60, 62.5, 65, 65, 65, 62.5, 60], [45, 50, 52.5, 55, 52.5, 50, 50, 50, 50, 50]], 'trumpetType': 'M', 'weartime': 21}
    # data =  {'age': 60, 'chip': 'e7111v2', 'chip_version': '2.1.50', 'company': 'onsemi', 'db': 1, 'gender': 1, 'lr': 0, 'mac': '40:D1:33:36:0F:01', 'outlook': 'CIC', 'phis': 1, 'series': 'Charm K830', 'srn': '126214726', 'tlt': [[40.0, 40.0, 45.0, 50.0, 50.0, 50.0, 55.0, 60.0, 60.0, 60.0], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], 'trumpetType': 'M', 'weartime': 365}
    data = {
                "age": 78, "chip": "e7111v2", "chip_version": "2.1.50",
                "company": "onsemi", "db": 1, "gender": 0, "lr": 0,
                "mac": "00:FF:7C:C6:ED:02", "outlook": "**CIC", "phis": 0,
                "series": "Charm E330", "srn": "1126143658",
                "tlt": [[80, 80, 80, 80, 82.5, 85, 85, 85, 95, 105], [-1] * 10],
                "trumpetType": "M", "weartime": 21
            }
    checkpoint = torch.load("../model/model.pt")
    model = YPNet()
    model.load_state_dict(checkpoint)
    model.eval()
    handler = RegressinHandler(model)

    # ===== 批量处理 =====
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    cases_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "test_cases.json")
    with open(cases_path, "r") as f:
        pool = json.load(f)
    end = min(len(pool), args.start + args.limit) if args.limit else len(pool)
    pool = pool[args.start:end]
    print(f"target batch {len(pool)} (start={args.start})")

    freqs_65 = [200,210,223,236,250,265,281,297,315,334,354,375,397,420,445,472,500,530,561,595,630,667,707,749,794,841,891,944,1000,
                1059,1122,1189,1260,1335,1414,1498,1587,1682,1782,1888,2000,2119,2245,2378,2520,2670,2828,2997,3175,3364,3564,3775,4000,
                4238,4490,4757,5040,5339,5657,5993,6350,6727,7127,7551,8000]
    results = []

    for i, case in enumerate(pool):
        cid = args.start + i
        print(f"[{i+1}/{len(pool)}] {cid} {case.get('mac','')} ...", end=" ")
        try:
            pre = handler.preprocess([{"data": json.dumps(case)}])
            pred = handler.inference(pre)
            pred_np = pred.cpu().detach().numpy().reshape(3, 65)
            curves = {
                "50": [round(float(v), 2) for v in pred_np[0]],
                "80": [round(float(v), 2) for v in pred_np[1]],
                "90": [round(float(v), 2) for v in pred_np[2]],
            }
            results.append({"id": cid, "curves": curves})
            print("OK")
        except Exception as e:
            results.append({"id": cid, "curves": None, "error": str(e)})
            print(f"FAIL: {e}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"target_results_{_run_suffix()}.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, "r") as exf:
            for r in json.load(exf).get("results", []):
                existing[r["id"]] = r
    for r in results:
        existing[r["id"]] = r
    merged = list(existing.values())
    with open(out_path, "w") as f:
        json.dump({"frequencies": freqs_65, "results": merged}, f, ensure_ascii=False, indent=2)
    ok = sum(1 for r in merged if r["curves"] is not None)
    print(f"  {ok}/{len(results)} OK, saved to {out_path}")
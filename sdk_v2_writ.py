import json
import os
import sys
import time
import argparse
import numpy as np
import uuid
import pprint
import random
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sdk"))
import sd

lib_mapping = {
    "E7111V2_2.1.50":"sdk\\CharmV2-0512-G630.library",
    "E7111V2_2.2.64":"sdk\\Condor.library",
    "E7111V2_2.7.201":"sdk\\F5300 2.2.7.library",
    "E7160SL_1.13.1507":"sdk\\Charm K830W.library",
    "E7160SL_1.18.1670":"sdk\\BTE Co 12W-118.library",
    "E7160SL_1.18.1670_RIC":"sdk\\RIC Charm W 1181670.library"
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
        return curves

    def set_mic_rec(self, mic, rec):
        model = self.library.TransducerModels
        res = model.GetByIdAndType(mic, sd.kMicrophoneSensitivityModel)
        self.product.SetTransducerModel(sd.kMicrophone1, res)
        res = model.GetByIdAndType(rec, sd.kReceiverSensitivityModel)
        self.product.SetTransducerModel(sd.kReceiver1, res)
        res = model.GetByIdAndType(rec, sd.kReceiverSaturationModel)
        self.product.SetTransducerModel(sd.kReceiver1, res)


API_URL = "http://192.168.110.55:50002/ai_yp_v3"
MIC_REC_MAPPING_FILE = os.path.join(os.path.dirname(__file__), "onsemi_mic_rec_mapping.json")
INVERSE_INDEX_FILE = os.path.join(os.path.dirname(__file__), "param_index", "onsemi_inverse_index_file.json")

with open(MIC_REC_MAPPING_FILE, "r", encoding="utf-8") as f:
    mic_rec_mapping = json.load(f)
with open(INVERSE_INDEX_FILE, "r", encoding="utf-8") as f:
    inverse_index = json.load(f)


def chip_display_to_sdk(chip, chip_version):
    """e7111v2 + 2.1.50 -> E7111V2_2.1.50"""
    _ver_map = {"1.18": "1.18.1670", "1.13": "1.13.1507"}
    full_ver = _ver_map.get(chip_version, chip_version)
    return f"E{chip[1:].upper()}_{full_ver}"


def lookup_mic_rec(series, outlook, chip_version, trumpet_type="M"):
    """?? series + outlook + chip_version + trumpetType ?? mic@rec ????"""
    prefix = "IDP_" + series.upper().replace(" ", "_").replace("-", "-")
    candidates = [f"{prefix}_{chip_version}", prefix, "default"]

    for key in candidates:
        if key not in mic_rec_mapping:
            continue
        entry = mic_rec_mapping[key]
        for ol_key in [outlook, "default"]:
            if ol_key not in entry:
                continue
            ol_val = entry[ol_key]
            if isinstance(ol_val, str):
                return ol_val, key
            if isinstance(ol_val, dict):
                for tk in [trumpet_type, "no_srn", "default"]:
                    if tk not in ol_val:
                        continue
                    val = ol_val[tk]
                    if isinstance(val, str):
                        return val, key
                    if isinstance(val, dict):
                        for sub_k in [trumpet_type, "default"]:
                            if sub_k in val:
                                return val[sub_k], key
    return "9446M@31570", "default"


def api_params_to_sdk_params(rl_params):
    """将 API 返回的编码转换为 SDK 参数名。"""
    sdk_params = {}
    for k, v in rl_params.items():
        name = inverse_index.get(str(k))
        if name is not None:
            sdk_params[name] = v
        else:
            print(f"[WARN] 未知参数编码: {k}")
    return sdk_params


def process_one(hipro, api_input, idx=0):
    """处理单条数据: 调用 API -> 转换参数 -> 写入 SDK -> 计算曲线。"""
    chip = api_input["chip"]
    chip_version = api_input["chip_version"]
    chip_sdk = chip_display_to_sdk(chip, chip_version)
    lr = "l" if api_input.get("lr", 0) == 0 else "r"
    outlook = api_input.get("outlook", "")
    series = api_input.get("series", "")

    # Step 1: lookup mic/rec
    try:
        mic_rec_str, mapping_key = lookup_mic_rec(series, outlook, chip_version, api_input.get("trumpetType", "M"))
    except Exception as e:
        raise RuntimeError(f"[step1-lookup_mic_rec] {e}")

    # Step 2: resolve library
    try:
        spec_chip = None
        if chip == "e7111v2" and mic_rec_str.split("@")[1] == "27926":
            lib_path = "sdk\\CharmV2-0512-B230.library"
            spec_chip = "E7111V2_2.1.50_HS_B230/E330"
        elif chip_sdk.startswith("E7160SL_1.18") and "CHARM" in mapping_key.upper():
            lib_path = "sdk\\RIC Charm W 1181670.library"
            chip_sdk = "E7160SL_1.18.1670_RIC"
        else:
            lib_path = lib_mapping.get(chip_sdk, list(lib_mapping.values())[0])
    except Exception as e:
        raise RuntimeError(f"[step2-resolve_lib] {e}")
    # Step 3: SDK init
    try:
        hipro.set_chip(chip_sdk)
        hipro.set_sdk(lib_path, lr)
        hipro.initial(spec_chip=spec_chip)
    except Exception as e:
        raise RuntimeError(f"[step3-sdk_init] {e}")

    # Step 4: set mic/rec
    try:
        mic, rec = mic_rec_str.split("@")[0], mic_rec_str.split("@")[1]
        hipro.set_mic_rec(mic, rec)
    except Exception:
        try:
            mic, rec = "9446M", "31570"
            hipro.set_mic_rec(mic, rec)
            mic_rec_str = "9446M@31570"
        except Exception as e:
            raise RuntimeError(f"[step4-set_mic_rec] {e}")

    # Step 5: E7160SL special init
    try:
        if chip_sdk == "E7160SL_1.18.1670":
            hipro.init_7160sl_118_params(mic_rec_str)
    except Exception as e:
        raise RuntimeError(f"[step5-E7160SL_init] {e}")

    # Step 6: call API
    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(api_input).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        rl_params = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
    except Exception as e:
        raise RuntimeError(f"[step6-call_api] {e}")

    # Step 7: convert params
    try:
        sdk_params = api_params_to_sdk_params(rl_params)
    except Exception as e:
        raise RuntimeError(f"[step7-convert_params] {e}")

    # Step 8: write to SDK
    try:
        hipro.write_parameters(0, sdk_params)
    except Exception as e:
        raise RuntimeError(f"[step8-write_params] {e}")

    # Step 9: get curve
    try:
        curve = hipro.get_curve()
    except Exception as e:
        raise RuntimeError(f"[step9-get_curve] {e}")

    print(f"[{idx}] {chip_sdk} {outlook} mic_rec={mic_rec_str} -> {len(rl_params)} params")
    return {"id": idx, "curves": curve}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量调用 API + SDK 计算频响曲线")
    parser.add_argument("--input", "-i", default="outputs/test_cases.json",
                        help="API 输入 JSON 文件路径（包含 api_input 数组）")
    parser.add_argument("--output", "-o", default="outputs/sdk_results.json",
                        help="结果输出 JSON 文件路径")
    parser.add_argument("--start", type=int, default=0,
                        help="从第几条数据开始处理（0-based）")
    parser.add_argument("--limit", type=int, default=0,
                        help="从 start 开始处理几条数据，0 表示全部到末尾")
    parser.add_argument("--log", action="store_true", help="同时保存终端输出到日志文件")
    args = parser.parse_args()

    if args.log:
        _ts = time.strftime("%Y%m%d%H%M")
        log_path = os.path.splitext(args.output)[0].replace("sdk_results", "log") + f"_{_ts}.txt"
        log_f = open(log_path, "w", encoding="utf-8")
        _builtin_print = print
        def print(*a, **kw):
            _builtin_print(*a, **kw)
            _builtin_print(*a, **kw, file=log_f)
        import atexit
        atexit.register(lambda: log_f.close())
    with open(args.input, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    total = len(test_cases)
    start = args.start
    end = start + args.limit if args.limit > 0 else total
    end = min(end, total)
    test_cases = test_cases[start:end]
    print(f"共 {total} 条数据，本次处理第 {start}~{end-1} 条（共 {len(test_cases)} 条）")

    hipro = HiproNoAids()
    results = []
    wrong_cases = []
    for i, case in enumerate(test_cases):
        try:
            result = process_one(hipro, case, idx=start + i)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] 第 {start + i} 条失败: {e}")
            results.append({"id": start + i, "error": str(e)})
            wrong_cases.append({
                "id": start + i,
                "error": str(e),
                "input": case,
            })

    timestamp = time.strftime("%Y%m%d%H%M")
    base, ext = os.path.splitext(args.output)
    output_path = f"{base}_{timestamp}{ext}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if wrong_cases:
        wrong_base, wrong_ext = os.path.splitext(args.output)
        wrong_path = wrong_base.replace("sdk_results", "wrong") + f"_{timestamp}{wrong_ext}"
        with open(wrong_path, "w", encoding="utf-8") as f:
            json.dump(wrong_cases, f, ensure_ascii=False, indent=2)
        print(f"其中 {len(wrong_cases)} 条失败，已保存至 {wrong_path}")

    print(f"\n处理完成，结果已保存至 {output_path}")


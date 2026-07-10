import sys, os, json, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sdk'))
import sd

# ========== 配置 ==========
api_input = {
    "age": 76, "chip": "e7160sl", "chip_version": "1.18",
    "company": "onsemi", "db": 1, "gender": 1, "lr": 1,
    "mac": "A4:6B:40:91:B8:1C", "outlook": "ITC", "phis": 2,
    "series": "Charm B230-W", "srn": "126140415",
    "tlt": [[75,80,75,65,75,75,75,80,60,60],[75,80,75,65,75,75,75,80,60,60]],
    "trumpetType": "M", "weartime": 365
}

API_URL = "http://192.168.110.55:50002/ai_yp_v3"
sdk_dir = os.path.join(os.path.dirname(__file__), 'sdk')
pm = sd.ProductManager()

# ========== Step 1: chip 映射 ==========
chip, chip_version = api_input['chip'], api_input['chip_version']
_ver_map = {"1.18": "1.18.1670", "1.13": "1.13.1507"}
chip_sdk = f"E{chip[1:].upper()}_{_ver_map.get(chip_version, chip_version)}"
lr = 'l' if api_input.get('lr', 0) == 0 else 'r'
print(f"Step1: chip_sdk={chip_sdk}, lr={lr}")
input("按 Enter 继续...")

# ========== Step 2: 选择 library ==========
lib_mapping = {
    "E7111V2_2.1.50": "sdk\\CharmV2-0512-G630.library",
    "E7111V2_2.2.64": "sdk\\Condor.library",
    "E7111V2_2.7.201": "sdk\\F5300 2.2.7.library",
    "E7160SL_1.13.1507": "sdk\\Charm K830W.library",
    "E7160SL_1.18.1670": "sdk\\BTE Co 12W-118.library",
    "E7160SL_1.18.1670_RIC": "sdk\\RIC Charm W 1181670.library"
}
lib_path = lib_mapping.get(chip_sdk, list(lib_mapping.values())[0])
print(f"Step2: lib_path={lib_path}")
input("按 Enter 继续...")

# ========== Step 3: SDK 初始化 ==========
with open(os.path.join(os.path.dirname(__file__), 'init_lib_params.json'), 'r') as f:
    init_params = json.load(f)
hipro_mem = 0
hipro = type('Hipro', (), {})()
hipro.mem = hipro_mem
hipro.library = pm.LoadLibraryFromFile(lib_path)
try:
    hipro.library.UnlockLibrary('89703236')
except: pass
hipro.product = hipro.library.Products.GetById(hipro_mem).CreateProduct()
# 写入 init_lib_params
params = init_params.get(chip_sdk, {})
for ptype in params:
    if 'SystemParameters' in ptype:
        prams = hipro.product.SystemMemory.Parameters
    else:
        prams = hipro.product.Memories[hipro_mem].Parameters
    for k, v in params[ptype].items():
        try:
            i = prams.GetById(k)
            if i.Type in {1,4,5}: i.Value = round(float(v))
            elif i.Type == 3: i.BooleanValue = round(float(v))
            elif i.Type == 2: i.DoubleValue = float(v)
        except: pass
print(f"Step3: SDK 初始化完成, chip_sdk={chip_sdk}")
input("按 Enter 继续...")

# ========== Step 4: 设置 mic/rec ==========
mic, rec = '9446M', '31570'
try:
    model = hipro.library.TransducerModels
    res = model.GetByIdAndType(mic, sd.kMicrophoneSensitivityModel)
    hipro.product.SetTransducerModel(sd.kMicrophone1, res)
    res = model.GetByIdAndType(rec, sd.kReceiverSensitivityModel)
    hipro.product.SetTransducerModel(sd.kReceiver1, res)
    res = model.GetByIdAndType(rec, sd.kReceiverSaturationModel)
    hipro.product.SetTransducerModel(sd.kReceiver1, res)
    print(f"Step4: mic/rec 设置成功: {mic}@{rec}")
except Exception as e:
    print(f"Step4 失败: {e}")
    # 尝试列出 library 中可用的 mic/rec
    model = hipro.library.TransducerModels
    print("  尝试其他 mic 名...")
    for test_mic in ['50PE30 MIC', '50PC33 MIC', '6950', 'EM-24446-CX MIC']:
        try:
            model.GetByIdAndType(test_mic, sd.kMicrophoneSensitivityModel)
            print(f"    [OK] {test_mic}")
        except: pass
input("按 Enter 继续...")

# ========== Step 5: 调用 API ==========
print("Step5: 调用 API...")
req = urllib.request.Request(API_URL, data=json.dumps(api_input).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
rl_params = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
print(f"Step5: API 返回 {len(rl_params)} 个参数")
input("按 Enter 继续...")

# ========== Step 6: 参数转换 ==========
with open(os.path.join(os.path.dirname(__file__), 'param_index', 'onsemi_inverse_index_file.json'), 'r') as f:
    inverse_index = json.load(f)
sdk_params = {}
for k, v in rl_params.items():
    name = inverse_index.get(str(k))
    if name: sdk_params[name] = v
print(f"Step6: 转换后 {len(sdk_params)} 个 SDK 参数")

# ========== Step 7: 写入 SDK ==========
mem = 0
prams = hipro.product.Memories[mem].Parameters
fail_count = 0
for k, v in sdk_params.items():
    try:
        i = prams.GetById(k)
        if i.Type in {1,4,5}: i.Value = round(float(v))
        elif i.Type == 3: i.BooleanValue = round(float(v))
        elif i.Type == 2: i.DoubleValue = float(v)
    except:
        fail_count += 1
        if fail_count <= 5:
            print(f"  [SKIP] {k}")
print(f"Step7: 写入完成, 跳过 {fail_count} 个无效参数")
input("按 Enter 继续...")

# ========== Step 8: 计算曲线 ==========
curves = {}
gd = hipro.product.Graphs.GetById(1)
gh = gd.CreateGraph()
st = gh.GraphSettings
freqs = [200,210,223,236,250,265,281,297,315,334,354,375,397,420,445,472,500,530,561,595,630,667,707,749,794,841,891,944,1000,1059,1122,
        1189,1260,1335,1414,1498,1587,1682,1782,1888,2000,2119,2245,2378,2520,2670,2828,2997,3175,3364,3564,3775,4000,4238,4490,4757,
        5040,5339,5657,5993,6350,6727,7127,7551,8000]
for label, level in [("50",50.0),("80",80.0),("90",90.0)]:
    if st.ContainsId("InputLevel"):
        st.GetById("InputLevel").DoubleValue = level
    gh.SetDomain(len(freqs), freqs)
    curves[label] = gh.CalculatePoints(len(freqs))
print(f"Step8: 计算完成, keys={list(curves.keys())}")
print("Done!")

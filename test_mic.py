import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sdk'))
import sd
pm = sd.ProductManager()
libs = ['BTE Co 12W-118.library','RIC Charm W 1181670.library','Charm K830W.library','CharmV2-0512-G630.library','Fascinat 5300W-118.library']
mics = ['9446M','50PE30 MIC','50PC33 MIC','6950','EM-24446-CX MIC']
for lib_name in libs:
    lib_path = os.path.join(os.path.dirname(__file__), 'sdk', lib_name)
    lib = pm.LoadLibraryFromFile(lib_path)
    try:
        lib.UnlockLibrary('89703236')
    except: pass
    models = lib.TransducerModels
    ok = []
    for mic in mics:
        try:
            models.GetByIdAndType(mic, sd.kMicrophoneSensitivityModel)
            ok.append(mic)
        except: pass
    print(f'{lib_name}: {ok}')

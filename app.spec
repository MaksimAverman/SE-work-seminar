# PyInstaller spec for the ICU Early Warning System desktop app.
# Build:  pyinstaller app.spec --noconfirm
from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_submodules

# --- bundled resources the app reads at runtime -------------------------------
datas = [
    ('images/eye.png', 'images'),
    ('images/eye_off.png', 'images'),
    ('images/male.png', 'images'),
    ('images/female.png', 'images'),
    ('backend/models/model.joblib', 'backend/models'),
    ('backend/models/scaler.joblib', 'backend/models'),
    ('backend/models/model_meta.json', 'backend/models'),
]
binaries = []
hiddenimports = []

# customtkinter ships its themes/assets as data files
datas += collect_data_files('customtkinter')

# lightgbm needs its native lib_lightgbm.dll
lgb_datas, lgb_binaries, lgb_hidden = collect_all('lightgbm')
datas += lgb_datas
binaries += lgb_binaries
hiddenimports += lgb_hidden

# make sure scikit-learn's lazily imported submodules are picked up
hiddenimports += collect_submodules('sklearn')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'seaborn', 'xgboost', 'shap', 'IPython', 'notebook',
        'torch', 'torchvision', 'numba', 'llvmlite', 'pyarrow',
        'tensorflow', 'transformers',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ICU_Early_Warning_System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,         # windowed GUI app (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

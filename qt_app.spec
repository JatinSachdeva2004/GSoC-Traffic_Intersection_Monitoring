# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    [r'D:\Downloads\finale6\khatam\qt_app_pyside\main.py'],
    pathex=['D:\Downloads\finale6\khatam'],
    binaries=[],
    datas=[(r'qt_app_pyside\\resources', r'qt_app_pyside\\resources'), (r'models/yolo11x_openvino_model', r'models/yolo11x_openvino_model'), (r'openvino_models', r'openvino_models'), (r'yolo11x_openvino_model', r'yolo11x_openvino_model'), (r'qt_app_pyside\\config.json', r'qt_app_pyside')],
    hiddenimports=['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],    name='traffic_monitoring_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


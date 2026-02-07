#!/usr/bin/env python
"""
Validation script to check if the refactored scripts are set up correctly
"""
import sys
import os


def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"  ✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"  ✓ Seaborn {sns.__version__}")
    except ImportError as e:
        print(f"  ✗ Seaborn import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"  ✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ✗ Scikit-learn import failed: {e}")
        return False
    
    return True


def check_custom_modules():
    """Check if custom modules can be imported"""
    print("\nChecking custom modules...")
    
    modules = [
        'custom_layers',
        'custom_metrics',
        'custom_callbacks',
        'model_builder',
        'data_utils'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} import failed: {e}")
            return False
    
    return True


def check_model_building():
    """Check if models can be built"""
    print("\nChecking model building...")
    
    try:
        from model_builder import build_writer_identification_model
        
        # Test each backbone
        backbones = ['resnet50', 'densenet201', 'xception', 'mobilenetv3']
        
        for backbone in backbones:
            try:
                model, _ = build_writer_identification_model(
                    backbone_name=backbone,
                    input_shape=(224, 224, 3),
                    num_clusters=64,
                    num_classes=179,
                    training_mode='frozen',
                    use_attention=False
                )
                print(f"  ✓ {backbone} model built successfully")
                del model  # Free memory
            except Exception as e:
                print(f"  ✗ {backbone} model build failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Model building check failed: {e}")
        return False


def check_files():
    """Check if all necessary files exist"""
    print("\nChecking files...")
    
    required_files = [
        'train.py',
        'batch_train.py',
        'model_builder.py',
        'data_utils.py',
        'custom_layers.py',
        'custom_metrics.py',
        'custom_callbacks.py',
        'requirements.txt',
        'README.md',
        '__init__.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            missing_files.append(file)
    
    return len(missing_files) == 0


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"  ✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"    - GPU {i}: {gpu.name}")
        else:
            print("  ⚠ No GPU found. Training will use CPU (slower).")
        
        return True
    except Exception as e:
        print(f"  ✗ GPU check failed: {e}")
        return False


def main():
    """Run all validation checks"""
    print("=" * 80)
    print("Validating Writer Identification Scripts Setup")
    print("=" * 80)
    
    checks = [
        ("Dependencies", check_imports),
        ("Custom Modules", check_custom_modules),
        ("Required Files", check_files),
        ("Model Building", check_model_building),
        ("GPU Availability", check_gpu),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n✓ All checks passed! The setup is ready to use.")
        print("\nYou can now run training:")
        print("  python train.py --help")
        print("\nOr see examples:")
        print("  python example_usage.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Make sure you're in the scripts/ directory")
        print("  - Check that all files were created correctly")
        return 1


if __name__ == '__main__':
    sys.exit(main())

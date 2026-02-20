#!/usr/bin/env python
"""
Environment Verification Script
Checks that all required packages are installed and working correctly.
Run this after setting up your environment to verify everything is ready.
"""

import sys

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally verify version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"⚠️  {package_name}: {version} (needs >= {min_version})")
                return False
        
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"⚠️  {package_name}: ERROR ({str(e)})")
        return False


def check_gpu():
    """Check if TensorFlow can see the GPU."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 0:
            print(f"\n✓ GPU Available: {len(gpus)} device(s) found")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
            
            # Try a simple computation
            try:
                x = tf.constant([1.0, 2.0, 3.0])
                y = x * 2
                result = y.numpy()
                print(f"✓ GPU Computation Test: PASSED (result: {result})")
                return True
            except Exception as e:
                print(f"✗ GPU Computation Test: FAILED")
                print(f"  Error: {str(e)}")
                return False
        else:
            print("\n✗ GPU Available: NO GPUs found")
            print("  TensorFlow will run on CPU only")
            return False
            
    except Exception as e:
        print(f"\n✗ GPU Check: ERROR ({str(e)})")
        return False


def main():
    """Run all verification checks."""
    print("="*80)
    print("Environment Verification for Muharaf Writer Identification")
    print("="*80)
    print()
    
    all_ok = True
    
    # Core packages
    print("Core Scientific Computing:")
    all_ok &= check_package("NumPy", "numpy", "1.21.0")
    all_ok &= check_package("Pandas", "pandas", "1.3.0")
    all_ok &= check_package("SciPy", "scipy", "1.5.0")
    all_ok &= check_package("scikit-learn", "sklearn", "0.24.0")
    all_ok &= check_package("h5py", "h5py")
    print()
    
    # Vision packages
    print("Image Processing & Computer Vision:")
    all_ok &= check_package("OpenCV", "cv2", "4.5.0")
    all_ok &= check_package("Pillow", "PIL", "8.0.0")
    print()
    
    # Visualization
    print("Visualization:")
    all_ok &= check_package("Matplotlib", "matplotlib", "3.4.0")
    all_ok &= check_package("Seaborn", "seaborn", "0.11.0")
    print()
    
    # Deep Learning
    print("Deep Learning Framework:")
    all_ok &= check_package("TensorFlow", "tensorflow", "2.8.0")
    all_ok &= check_package("Keras", "keras")
    print()
    
    # Utilities
    print("Utilities:")
    all_ok &= check_package("tqdm", "tqdm")
    all_ok &= check_package("PyYAML", "yaml")
    all_ok &= check_package("requests", "requests")
    print()
    
    # GPU check
    print("GPU Support:")
    gpu_ok = check_gpu()
    print()
    
    # Summary
    print("="*80)
    if all_ok and gpu_ok:
        print("✓ ALL CHECKS PASSED - Environment is ready!")
        print()
        print("Next steps:")
        print("  cd scripts")
        print("  python train.py --backbone resnet50 --training-mode frozen --epochs 1 --batch-size 32")
        return 0
    elif all_ok and not gpu_ok:
        print("⚠️  PACKAGES OK, BUT GPU NOT AVAILABLE")
        print()
        print("You can still train on CPU (slower):")
        print("  CUDA_VISIBLE_DEVICES='' python train.py ...")
        print()
        print("Or fix GPU support:")
        print("  pip uninstall tensorflow")
        print("  pip install tensorflow[and-cuda]")
        return 1
    else:
        print("✗ SOME PACKAGES MISSING OR OUTDATED")
        print()
        print("Fix by running:")
        print("  pip install tensorflow[and-cuda]")
        print("  pip install -r scripts/requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

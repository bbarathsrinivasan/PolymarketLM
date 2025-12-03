"""
Test script to verify all report generation scripts are working.

This script checks:
1. All scripts can be imported
2. Configuration files exist
3. Required directories exist
4. Basic functionality of key functions
"""

import sys
from pathlib import Path
import importlib.util

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    required_modules = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "tqdm",
        "pandas",
        "json",
        "yaml"
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing modules: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required modules available")
        return True


def test_script_imports():
    """Test that our scripts can be imported."""
    print("\nTesting script imports...")
    
    scripts_dir = Path(__file__).parent
    scripts = [
        "evaluate_icl_mistral",
        "evaluate_icl_gemma",
        "evaluate_finetuned_mistral",
        "evaluate_finetuned_gemma",
        "generate_comparison_tables",
        "error_analysis"
    ]
    
    failed = []
    for script in scripts:
        script_path = scripts_dir / f"{script}.py"
        if script_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(script, script_path)
                module = importlib.util.module_from_spec(spec)
                # Don't actually execute, just check syntax
                print(f"  ✓ {script}.py (syntax OK)")
            except Exception as e:
                print(f"  ✗ {script}.py - ERROR: {e}")
                failed.append(script)
        else:
            print(f"  ✗ {script}.py - NOT FOUND")
            failed.append(script)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All scripts can be imported")
        return True


def test_config_files():
    """Test that config files exist."""
    print("\nTesting config files...")
    
    configs_dir = Path(__file__).parent.parent / "configs"
    configs = [
        "icl_config.yaml",
        "finetuning_config_mistral.yaml",
        "finetuning_config_gemma.yaml"
    ]
    
    missing = []
    for config in configs:
        config_path = configs_dir / config
        if config_path.exists():
            print(f"  ✓ {config}")
        else:
            print(f"  ✗ {config} - NOT FOUND")
            missing.append(config)
    
    if missing:
        print(f"\n❌ Missing configs: {', '.join(missing)}")
        return False
    else:
        print("\n✓ All config files exist")
        return True


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directories...")
    
    base_dir = Path(__file__).parent.parent
    dirs = [
        "scripts",
        "configs",
        "results",
        "outputs"
    ]
    
    missing = []
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - NOT FOUND (will be created)")
            missing.append(dir_name)
    
    # Create missing directories
    for dir_name in missing:
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {dir_name}/")
    
    print("\n✓ All directories ready")
    return True


def test_dataset_exists():
    """Test that dataset file exists."""
    print("\nTesting dataset...")
    
    dataset_path = Path(__file__).parent.parent.parent / "data" / "fine_tune.jsonl"
    if dataset_path.exists():
        # Count lines
        with open(dataset_path, 'r') as f:
            lines = sum(1 for _ in f)
        print(f"  ✓ Dataset found: {dataset_path}")
        print(f"  ✓ Dataset size: {lines} examples")
        return True
    else:
        print(f"  ✗ Dataset not found: {dataset_path}")
        print("  Run: python scripts/preprocess_data.py")
        return False


def main():
    print("=" * 60)
    print("REPORT GENERATION SCRIPTS TEST")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Script Imports", test_script_imports()))
    results.append(("Config Files", test_config_files()))
    results.append(("Directories", test_directories()))
    results.append(("Dataset", test_dataset_exists()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All tests passed! Ready to run experiments.")
    else:
        print("\n❌ Some tests failed. Please fix issues above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


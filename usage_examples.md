# Usage Examples for complete_comparison.py

## Basic Usage
```bash
# Run with default settings (k=1,2,3, logical error calculation DISABLED by default)
python complete_comparison.py
```

## Skip Distance Calculation
```bash
# Skip circuit distance calculation and plotting (faster)
python complete_comparison.py --skip-distance
```

## Enable Logical Error Rate Calculation
```bash
# Enable logical error rate calculation and plotting (DISABLED by default)
python complete_comparison.py --enable-logical-error
```

## Skip Logical Error Rate Calculation (explicit)
```bash
# Explicitly skip logical error rate calculation and plotting
python complete_comparison.py --skip-logical-error
```

## Skip Crumble URL Generation
```bash
# Skip Crumble URL generation (slightly faster)
python complete_comparison.py --skip-crumble
```

## Custom k Values
```bash
# Test only k=1 and k=2
python complete_comparison.py --k-values 1 2
```

## Custom Shots
```bash
# Use fewer shots for faster testing
python complete_comparison.py --shots 10000
```

## Custom Noise Levels
```bash
# Test specific noise levels
python complete_comparison.py --noise-levels 0.001 0.002 0.005
```

## Fast Testing (Minimal Analysis)
```bash
# Just create circuits and generate Crumble URLs, skip all calculations
python complete_comparison.py --skip-distance --shots 1000
```

## Full Analysis with Custom Parameters
```bash
# Full analysis with custom parameters (enable logical error calculation)
python complete_comparison.py --enable-logical-error --k-values 1 2 3 4 --shots 100000 --noise-levels 0.0001 0.001 0.01
```

## Available Options
- `--k-values`: Space-separated list of k values to test (default: 1 2 3)
- `--shots`: Number of shots for logical error rate calculation (default: 10,000,000)
- `--noise-levels`: Space-separated list of physical error rates to test (default: logspace from 0.000316 to 0.01)
- `--skip-distance`: Skip circuit distance calculation and plotting
- `--skip-logical-error`: Skip logical error rate calculation and plotting (default: True)
- `--enable-logical-error`: Enable logical error rate calculation and plotting (overrides --skip-logical-error)
- `--skip-crumble`: Skip Crumble URL generation

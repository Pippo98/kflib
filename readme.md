# Kalman Filter Library (kflib)

A C++ library implementing various Kalman filter variants with Eigen-based linear algebra.

## Repository Structure
- `include/kflib/` - Public header files
- `src/` - Implementation source files
- `examples/` - Example applications:
  - `example_kf.cpp` - Standard Kalman Filter demo
  - `example_ekf.cpp` - Extended Kalman Filter demo
  - `example_ukf.cpp` - Unscented Kalman Filter demo
- `eigen/` - Eigen3 linear algebra library (submodule)

## Getting Started

### Cloning the Repository
```bash
# Clone with submodules (required for Eigen dependency)
git clone --recursive https://github.com/Pippo98/kflib.git
```

### Building the Project
```bash
# Standard build
cd kflib
cmake -Bbuild
cmake --build build

# Build with examples enabled
cmake -Bbuild -DKFLIB_BUILD_EXAMPLES=ON
cmake --build build
```

## Example Usage
After building with examples enabled:
```bash
# Run standard Kalman Filter example
./build/example_kf

# Run Extended Kalman Filter example
./build/example_ekf

# Run Unscented Kalman Filter example
./build/example_ukf
```

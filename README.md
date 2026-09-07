This code offers an implementation of Modulus of continuity computation by exploiting Projected Gradient Descent algorithm. Current code supports L1, L2 norm (d_x).

Code is reported in pgd_moc.py file, whereas a comparison with some analytical moc and the discrete modulus of continuity (DMOC) is presented in main.py



## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)

## Installation
It is recommended to create a virtual environment before installing dependencies.
suggested python >=3.14

## Create a virtual environment
```bash
python -m venv venv
pip install -r requirements.txt
```


# FMCA

## Ensure OpenMP is available in your compiler configured in CMAKE (e.g. g++)

```bash
export CC=/opt/homebrew/bin/gcc-16  
export CXX=/opt/homebrew/bin/g++-16
```


## Install pybind11 (sugg. via homebrew )

```bash
brew install pybind11
```

## Install Eigen (sugg. via homebrew)

```bash
brew install eigen
```


## Install FMCA and set the DD branch

```bash
git clone https://github.com/muchip/fmca.git
cd fmca
git checkout DD 
```


## Compile FMCA (following FMCA README.md)  (later we might use our cmake file)
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make    
cd .. 
```  


## Usage
Grant permission and run the .sh script for reproducing paper results
```bash
chmod +x run.sh
./run.sh
```  

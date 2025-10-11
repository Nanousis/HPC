# Installation instructions for the intel compiler
```
# add Intel GPG key
sudo wget -qO /etc/apt/trusted.gpg.d/intel-oneapi.asc \
  https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

# add the oneAPI apt repository
echo "deb https://apt.repos.intel.com/oneapi all main" | \
  sudo tee /etc/apt/sources.list.d/intel-oneapi.list
```

```
sudo apt update

# install the DPC++/C++ compiler package (contains icx/icpx)
sudo apt install intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic
```

adding this to bash `source /opt/intel/oneapi/setvars.sh` 
```
echo -e '\n# Intel oneAPI environment\nif [ -f /opt/intel/oneapi/setvars.sh ]; then\n    source /opt/intel/oneapi/setvars.sh > /dev/null\nfi' >> ~/.bashrc
source ~/.bashrc
```
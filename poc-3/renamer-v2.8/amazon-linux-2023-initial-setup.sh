dnf update -y
dnf install -y gcc gcc-c++ make autoconf automake libtool pkgconfig
dnf install -y libpng-devel libjpeg-devel libtiff-devel zlib-devel
dnf groupinstall -y "Development Tools"
dnf install -y python3-devel mesa-libGL-devel
dnf install -y make gcc
dnf install -y iproute
dnf install -y unzip
dnf install -y mc
dnf install -y nano
dnf install -y wget
dnf install -y git
dnf install python3.12

# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# # bash Miniconda3-latest-Linux-x86_64.sh
# curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# bash Miniforge3-Linux-aarch64.sh

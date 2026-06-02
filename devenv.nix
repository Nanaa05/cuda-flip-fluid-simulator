{ pkgs, ... }:

{
  packages = with pkgs; [
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cudart
    cudaPackages.nsight_systems
    cudaPackages.nsight_compute
    
    libx11
    libGL
    libGLU
    freeglut

    python3
    python3Packages.matplotlib
    python3Packages.numpy
  ];

  languages.python = {
    enable = true;
  };

  env = {
    CUDA_PATH = "${pkgs.cudaPackages.cuda_nvcc}";
    
    LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.libGL}/lib:${pkgs.libx11}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:\$LD_LIBRARY_PATH";
  };

  enterShell = ''
    echo "CUDA Environment Ready"
    nvcc --version
    echo "Nsight Profilers Ready"
    nsys --version
    echo "Python Data Visualization Ready"
    python3 --version
  '';
}

{ pkgs, ... }:

{
  # Pack up your compilation toolkit
  packages = with pkgs; [
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cudart
    
    xorg.libX11
    libGL
    libGLU
    freeglut
  ];

  # Map system paths dynamically 
  env = {
    CUDA_PATH = "${pkgs.cudaPackages.cuda_nvcc}";
    LD_LIBRARY_PATH = "${pkgs.libGL}/lib:${pkgs.xorg.libX11}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:\$LD_LIBRARY_PATH";
  };

  enterShell = ''
    echo "CUDA Environment Ready"
    nvcc --version
  '';
}

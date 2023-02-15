{ pkgs ? import <nixpkgs> { } }:

with pkgs;
with python310Packages;
let
  py3dmol = buildPythonPackage rec {
      pname = "py3Dmol";
      version = "1.8.1";
      src = fetchPypi {
        pname = "${pname}";
        version = "${version}";
        format = "setuptools";
        sha256 = "sha256-ELswIsXbPl5tZvUdfUEYr6+TlybWVfCHYXCd3wMssx0=";
      };
    };
  pyrdtp = buildPythonPackage rec {
      pname = "pyRDTP";
      version = "0.2";
      src = pkgs.fetchFromGitLab {
        owner = "iciq-tcc/nlopez-group";
        repo = "${pname}";
        rev = "v${version}";
        sha256 = "sha256-NAjKEhgGLzKtlRs5xiv9nTrF9B08upuGQB2E25oLVjM=";
      };
      doCheck = false;
      propagatedBuildInputs = [
        py3dmol
        numpy
        scipy
        shapely
        networkx
        matplotlib
        setuptools
        pandas
        scikitlearn
      ];
      meta = with pkgs.lib; {
        description = "pyRDTP is a Python library to manipulate molecules";
        homepage = "https://gitlab.com/iciq-tcc/nlopez-group/pyrdtp";
        license = licenses.mit;
      };
    };
  pyg-lib = buildPythonPackage rec {
    name = "pyg_lib";
    version = "0.1.0";
    format = "wheel";
    src = fetchurl {
      url = "https://data.pyg.org/whl/torch-1.12.0%2Bcpu/${name}-${version}%2Bpt112cpu-cp310-cp310-linux_x86_64.whl";
      sha256 = "sha256-T7Ejq7Nuc3y7Iiq+F6BraNNP8hiKMbX6MJ7lR6eQn3w=";
    };

    doCheck = true;

    propagatedBuildInputs = with python310Packages; [
      numpy
      scipy
      boost
    ];
  };
  torch-sparse = buildPythonPackage rec {
      pname = "torch_sparse";
      version = "0.6.16";

      src = pkgs.python310.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-dlfP3KwiH7kqoAr35gMoU6fSzA1YFR2UNkhQlKt2Naw=";
      };

      # no tests because this is a simple example
      doCheck = false;

      nativeBuildInputs = [ pkgs.which ];
      propagatedBuildInputs = [ scipy
                                pytorch
                                future
                                pybind11
                                pytest
                                pytestrunner
                                pytest-flakes
                              ];
    };
  torch-cluster = buildPythonPackage rec {
      pname = "torch_cluster";
      version = "1.6.0";

      src = pkgs.python310.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-JJwb2MM6iHsiv1aaWdCGhUWAQDISNZTdjHa6GIWFnDk=";
      };

      # no tests because this is a simple example
      doCheck = false;

      nativeBuildInputs = [ pkgs.which ];
      propagatedBuildInputs = [ scipy pytorch future pybind11 ];
    };
  torch-scatter = buildPythonPackage rec {
      pname = "torch_scatter";
      version = "2.0.9";

      src = pkgs.python310.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-CPVRHWRHO63wpx0VazbcKwm5wvAKfNNzuTW0kMR3p/E=";
      };

      # no tests because this is a simple example
      doCheck = false;

      nativeBuildInputs = [ pkgs.which ];
      propagatedBuildInputs = [ scipy pytorch future pybind11 ];
    };
  torch-spline-conv = buildPythonPackage rec {
      pname = "torch_spline_conv";
      version = "1.2.1";

      src = pkgs.python310.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-Nk9ljg7LTFJjpyjClhVT4CL8RMEaYz1aG/mGzxaatDg=";
      };

      # no tests because this is a simple example
      doCheck = false;

      nativeBuildInputs = [ pkgs.which ];
      propagatedBuildInputs = [ scipy
                                pytorch
                                future
                                pybind11
                                pytest
                                pytestrunner
                                pytest-flakes
                              ];
    };
  pyg = buildPythonPackage rec {
      pname = "pytorch_geometric";
      version = "a7e6be4";
      format = "setuptools";
      src = pkgs.fetchFromGitHub {
        owner = "pyg-team";
        repo = "${pname}";
        rev = "a7e6be49e9a5e9e3ca5cedcdcb37994a32e38335";
        sha256 = "sha256-HV4VXsykSENv+eCRm5M7VG5FKxNcYj2iC0Y4w/ZKZcE=";
      };
      doCheck = false;
      propagatedBuildInputs = [
        scipy
        scikitlearn
        tqdm
        jinja2
        pyparsing
        torch
        torch-cluster
        torch-sparse
        torch-scatter
        torch-spline-conv
        pyg-lib
      ];
    };
  ocp = buildPythonPackage rec {
      pname = "ocp";
      version = "7d1040bddc590dc70790493b802f0ef6dd968d95";
      format = "setuptools";
      src = pkgs.fetchFromGitHub {
        owner = "Open-Catalyst-Project";
        repo = "${pname}";
        rev = "7d1040bddc590dc70790493b802f0ef6dd968d95";
        sha256 = "sha256-ODTFtUXTMQhYQa1P0QIfLT7c0pE5YYlWS6CTJUFh0bU=";
      };
      doCheck = false;
      propagatedBuildInputs = [
        pyg
        ase
        matplotlib
        pre-commit
        pymatgen
        pyyaml
        tensorboard
        tqdm
        sphinx
        nbsphinx
        pandoc
        black
        numba
        wandb
        lmdb
        pytest
        sphinx-rtd-theme
        ipywidgets
        pyyaml
      ];
    };

in pkgs.mkShell {
  buildInputs = with pkgs; [ pyrdtp ocp jupyterlab pandas seaborn ];
  shellHook = ''
    export PYTHONPATH=${builtins.getEnv "PWD"}:$PYTHONPATH
  '';
}

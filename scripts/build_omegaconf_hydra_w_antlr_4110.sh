
# TODO: temporary fixes for omegaconf/hydra to use antlr 4.11.0 (needed by parse_latex for training on hendrycks math dataset)

# Check if Java is installed, if not install OpenJDK 17
if ! command -v java &> /dev/null
then
    echo "Java could not be found, installing OpenJDK 17..."
    sudo apt update
    sudo apt install -y openjdk-17-jdk
fi


pip uninstall omegaconf hydra-core

# clone sources
git clone https://github.com/omry/omegaconf
git clone https://github.com/facebookresearch/hydra

# checkout specific versions (Hydra 1.3.2 + OmegaConf 2.3.0)
(
  cd omegaconf
  git checkout v2.3.0
)
(
  cd hydra
  git checkout v1.3.2
)

# download the antlr 4.11.0 binary:
wget https://www.antlr.org/download/antlr-4.11.0-complete.jar

# replace the old antlr binaries in BOTH omegaconf and hydra
rm omegaconf/build_helpers/bin/antlr-4.9.3-complete.jar
rm hydra/build_helpers/bin/antlr-4.9.3-complete.jar
cp antlr-4.11.0-complete.jar omegaconf/build_helpers/bin/
cp antlr-4.11.0-complete.jar hydra/build_helpers/bin/

# update all references to 4.9.3 / 4.9.* to 4.11.0 in omegaconf + hydra only
(
  cd omegaconf
  grep -ErlI '4\.9\.[\*3]' . | xargs sed -E -i 's/4\.9\.[\*3]/4.11.0/g' || true
)
(
  cd hydra
  grep -ErlI '4\.9\.[\*3]' . | xargs sed -E -i 's/4\.9\.[\*3]/4.11.0/g' || true
)

pip install "antlr4-python3-runtime==4.11.0"

pip install build

(
  cd omegaconf
  python -m build
)
(
  cd hydra
  python -m build
)

pip install omegaconf/dist/omegaconf-2.3.0-*.whl
pip install hydra/dist/hydra_core-1.3.2-*.whl

# verify that the Python runtime installed is 4.11.0
pip list 2>/dev/null | grep antlr4

# cleanup
rm -rf omegaconf hydra antlr-4.11.0-complete.jar
#bash scripts/build_omegaconf_hydra_w_antlr_4110.sh
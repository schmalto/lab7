LIGHT_GREEN='\033[1;32m'
NC='\033[0m' # No Color
LIGHT_CYAN='\033[1;36m'
RED='\033[0;31m'



if [ $EUID != 0 ]; then
    sudo "$0" "$@"
    exit $?
fi


echo "${LIGHT_CYAN}[Setup]${NC} Installing Github CLI"

type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

echo "${LIGHT_CYAN}[Setup]${NC} Installed Github CLI"
echo "${LIGHT_CYAN}[Setup]${NC} Installing Libraries"
pip install scikit-learn
pip install matplotlib
pip install librosa
pip install numpy
pip install scikit-image
pip install scipy
# pip install ipykernel
echo "${LIGHT_CYAN}[Setup]${NC} Installed all Libraries"
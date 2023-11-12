echo "Creating vertual environment"

conda create -p venv python=3.9 -y

echo "Environment created"

echo "---------------------------------"
echo "Activating the environment"

conda activate venv/

echo "Activated the environment"
echo "----------------------------------"

echo "installing the requirements"

pip install -r requirements.txt

echo "installation completed"
echo "----------------------------------"





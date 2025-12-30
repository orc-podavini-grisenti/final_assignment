## 🚀 Getting Started

Follow these steps to set up the development environment and run the simulation.

### 1. Set Up the Environment
It is recommended to use a virtual environment to keep dependencies isolated.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate   # On Linux/macOS:
.venv\Scripts\activate    # On Windows:
``` 

### 2. Install Dependencies
Install the required Python packages specified in requirements.txt.
```bash
pip install -r requirements.txt
```

### 3. Running the Simulation
Note: This project is structured to be run from the root directory.
Before running any scripts, ensure the root directory is added to your Python path so imports work correctly.

```bash
# 1. Export the Python path (Run this once per terminal session)
export PYTHONPATH=$PYTHONPATH:.
# 2. Run the scripts from the root folder
python planner/execute_dubins.py # Example: Running the Dubins Path Executor
```
Tip: Alternatively, you can run scripts as modules without manually exporting the path: 
```bash
python -m planner.execute_dubins
```
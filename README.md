<img src="docs/figs/logo.png" height="100px">

CIBUSmod is a food systems model developed at the [Department of Energy and Technology, Swedish University of Agricultural Sciences (SLU)](https://www.slu.se/en/departments/energy-technology/) within the [MISTRA Food Futures Programme](https://mistrafoodfutures.se/).

To get started look at the [users guide](docs/MANUAL.md) for some guidance on how to run the model and work with scenarios and [this notebook](notebooks/run_scn.ipynb) for an example run of the model and some outputs.

# Installing and running CIBUSmod (Windows)
These installation instructions relies on working installations of [Python](https://www.python.org/downloads/windows/) and [Git](https://git-scm.com/download/win).

Open a new `Command Prompt` and `cd` to the directory where you want to place CIBUSmod and download the CIBUSmod code from github.

```
git clone https://github.com/SLU-foodsystems/CIBUSmod
cd CIBUSmod
```
Next, create and activate a new virtual environment to keep Python packages needed for CIBUSmod separate from your base Python installation.
```
python -m venv .venv --clear --upgrade-deps --prompt 'CIBUSmod-venv'
.venv\Scripts\activate
```
After activating the virtual environment, make sure that you see `('CIBUSmod-venv')` at the beginning of the command line, which indicates that the virtual environment is active. Now it´s time to install all python packages needed to run CIBUSmod and start jupyter lab.
```
pip install --upgrade pip
pip install --require-virtualenv -r requirements.txt
ipython kernel install --user --name="CIBUSmod-venv"
jupyter lab
```
Once jupyter lab is started, navigate to the `notebooks` folder and open one of the notebooks. Make sure that the `CIBUSmod-venv` kernel is selected via `Kernel > Change kernel > CIBUSmod-venv`. Run the notebook. That's it!

After quitting jupyter lab and returning to the `Command Prompt`, type `deactivate`, or simply close the `Command Prompt`, to exit the virtual environment,

Next time, you open a new `Command Prompt`, `cd` to the `CIBUSmod` directory and activate the virtual invironment before starting `jupyter lab`.
```
.venv\Scripts\activate
jupyter lab
```

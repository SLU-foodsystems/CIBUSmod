<img src="docs/figs/logo.png" height="100px">

CIBUSmod is a food systems model currently under development at the Department of Energy and Technology, [Swedish University of Agricultural Sciences (SLU)](https://www.slu.se/en/) within the [MISTRA Food Futures Programme](https://mistrafoodfutures.se/).

Check out [this notebook](https://github.com/SLU-foodsystems/CIBUSmod/blob/main/notebooks/run_scn.ipynb) for an example run of the model and some outputs. This reflects the model in it's current state and will be continuosly updated as the work progresses.

# Installing and running CIBUSmod (Windows)
Open a new `Command Prompt` and run the commands below. This will download the CIBUSmod code, create and activate a new virtual environment, install all python packages needed to run the model, and finally start jupyter notebook.

```
cd <directory to place CIBUSmod>
git clone https://github.com/SLU-foodsystems/CIBUSmod
cd CIBUSmod
python -m venv .venv --clear --upgrade-deps --prompt 'CIBUSmod-venv'
.venv\Scripts\activate
pip install "pandas<2" scipy==1.10 matplotlib cvxpy geopandas openpyxl jinja2 ipykernel
ipython kernel install --user --name="CIBUSmod-venv"
jupyter notebook
```
Once jupyter notebook is started, navigate to the `notebooks` folder and open one of the notebooks. Make sure that the `CIBUSmod-venv` kernel is selected via `Kernel > Change kernel > CIBUSmod-venv`. Run the notebook. That's it!

After quiting jupyter notebook and returning to the `Command Prompt`, type `deactivate` to exit the virtual environment.

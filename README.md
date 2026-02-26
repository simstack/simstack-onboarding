# Simstack Onboarding
Repo for the onboarding and introduction to Simstack II


## Access

1. Register your account at: https://simstack.int.kit.edu
2. Register access to the jupyterhub: https://material-digital.de/
3. Go to: https://pyiron.material-digital.de/

Some jupyter instance should pop up

## Installation 

1. In the jupyter instance go to "+" and open a terminal
2. Install uv https://docs.astral.sh/uv/getting-started/installation/#installation-methods
3. Add it to your shell: source $HOME/.local/bin/env 
4. Clone this repo: git clone https://github.com/simstack/simstack-onboarding.git
5. In simstack-onboarding run uv sync --locked 
6. In the home directory make a directory simstack: mkdir simstack
7. go to simstack-onboarding 
8. create a simstack.toml file (on the left in the jupyter instance, navigate to simstack-onboarding, click on "new file" in the file folder):
9. uv run create_model_table --dir private --dir public
10. uv run create_node_table --dir private --dir public_

```toml
[parameters]
[parameters.general]
use_db = true
workdir_self = "PATH TO YOUR WORKDIR"
# these are parameters for one user for all hosts
[parameters.db]
database = "USERNAME_data"
test_database = "USERNAME_test"
connection_string="CONNECTION STRING"
```







We need a virtual machine with N users, each user should have uv installed / ALternatively users can run on their laptops
We need databases for each user, i would preinstall the models and nodes 
The users must set up a runner 

## Installation

1. Fork this repo
2. Clone your fork
3. Make a simstack.toml





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
4. Clone/fork this repo: git clone https://github.com/simstack/simstack-onboarding.git (fork if you want to commit changes)
5. In simstack-onboarding run uv sync --locked 
6. In the home directory make a directory simstack: mkdir simstack
7. go to simstack-onboarding 
8. create a simstack.toml file (on the left in the jupyter instance, navigate to simstack-onboarding, click on "new file" in the file folder):
9. uv run create_model_table --dir private --dir public
10. uv run create_node_table --dir private --dir public
11. export PYTHONPATH=`pwd` (note the backslash, add to your .bashrc if you want it to be permanent)
11. uv run simstack_runner --resource local --no-pull

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
## Running the first workflow

1. in the UI go to submit 
2. under public / first_steps / simple_operations / add click "favorite"
3. select "favorite"
4. fill in some numbers for a and b 
5. click submit worklow 
6. in the task tabs you should see an entry "add", first yellow, then hopefully green
7. click on the green entry and you should see the inputs / outputs / logs of the workflow
8. in the jupyter instance lots of log entries should have been printed to the terminal
8. go to the jupyter instance and create a new terminal, go to the simstack directory
9. you should see a new folder called "add" with an empty directory with the job_id


## Writing your first workflow

1. inspect the code in the first_steps/simple_operations.py
2. copy this file to first_steps/complex_operations.py
3. write a function complex_operation which takes three args, adds the first two using the sum node and multiplies the result with the third arg
4. test the function in the jupyter instance by adding the following code 

```python
import asyncio 
from simstack.core.context import context 

async def main():
    await context.initialize()
    arg1 = FloatData(field_name="arg1",value=1)
    arg2 = FloatData(field_name="arg2",value=2)
    arg3 = FloatData(field_name="arg3",value=3)
    result = await complex_operation(arg1,arg2,arg3)
    print(result.value)

if __name__ == "__main__":
    asyncio.run(main())
```

5. open a terminal and set  export PYTHONPATH=`pwd` run this code in a terminal as:  uv run python public/first_steps/complex_workflow.py. You should get an error like this: 


```aiignore
2026-02-26 11:33:50 - Context         - INFO       - context.py          : 157 - Database connection to None mongodb://wolfgang-onboarding@simstack.int.kit.edu:27017//None
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    :  75 - toml-file read, use_db_for_init: True
2026-02-26 11:33:50 - init_resource   - INFO       - init_data_source.py :  18 - Initializing ConfigReader from database, allowed resources: ['local', 'self']
2026-02-26 11:33:50 - init_resource   - INFO       - init_data_source.py :  32 - ConfigReader resources: ['local', 'self']
2026-02-26 11:33:50 - init_resource   - WARNING    - init_data_source.py :  44 - Initializing paths from database is not yet implemented
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 103 - ConfigReader initialized with workdir: /home/jovyan/simstack
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.resource_str: self
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.workdir: /home/jovyan/simstack
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.hostname: localhost
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.python_paths: []
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.environment_start: None
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.ssh_key: None
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.routes: []
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.is_default: False
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.git_branch: main
2026-02-26 11:33:50 - config-reader   - INFO       - config_reader.py    : 105 - resource_definition.id: 69a02f9edf0b43d7cbef6ef3
2026-02-26 11:33:50 - Node            - ERROR      - node.py             : 183 - Could not find function mapping for name: complex_operation
Traceback (most recent call last):
  File "/home/jovyan/simstack-onboarding/public/first_steps/complex_workflow.py", line 21, in <module>
    asyncio.run(main())
  File "/home/jovyan/simstack-onboarding/.venv/lib/python3.12/site-packages/nest_asyncio.py", line 30, in run
    return loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/simstack-onboarding/.venv/lib/python3.12/site-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
           ^^^^^^^^^^
  File "/home/jovyan/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/asyncio/futures.py", line 202, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/home/jovyan/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/asyncio/tasks.py", line 314, in __step_run_and_handle_result
    result = coro.send(None)
             ^^^^^^^^^^^^^^^
  File "/home/jovyan/simstack-onboarding/public/first_steps/complex_workflow.py", line 17, in main
    result = await complex_operation(arg1,arg2,arg3)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/simstack-onboarding/.venv/lib/python3.12/site-packages/simstack/core/node.py", line 889, in sync_wrapper
    status = loop.run_until_complete(execution_node.get_node_registry())
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jovyan/simstack-onboarding/.venv/lib/python3.12/site-packages/nest_asyncio.py", line 98, in run_until_complete
    return f.result()
           ^^^^^^^^^^
  File "/home/jovyan/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/asyncio/futures.py", line 202, in result
    raise self._exception.with_traceback(self._exception_tb)
  File "/home/jovyan/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/asyncio/tasks.py", line 314, in __step_run_and_handle_result
    result = coro.send(None)
             ^^^^^^^^^^^^^^^
  File "/home/jovyan/simstack-onboarding/.venv/lib/python3.12/site-packages/simstack/core/node.py", line 267, in get_node_registry
    await self.make_registry_entry(function_hash, arg_hash)
  File "/home/jovyan/simstack-onboarding/.venv/lib/python3.12/site-packages/simstack/core/node.py", line 184, in make_registry_entry
    raise ValueError(f"Could not find function mapping for name: {self.name}")
ValueError: Could not find function mapping for name: complex_operation

```

the first part tells you that the process started OK, but the second part tells you that the function complex_operation is not registered.

6. rerun: uv run create_node_table --dir private --dir public and the program again (should work now)
7. go to the UI
8. inspect the task tab, click on the little arrow next to the complex_operation task and should see the subtasks
9. go to submit: click on the complex_operation task, fill in some values and click submit
10. go to the task tab and inspect the results 



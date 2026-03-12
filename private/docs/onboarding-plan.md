# Onboarding Plan

1) Introduction (WW, Powerpoint, ca. 10-15 min)
2) UI Intro & First Steps (Artem, ca. 20-30 min)
3) Machine Learning (Jörg, ca. 20-30 min)
4) Future Plans (WW, ca. 10 min)
5) Q&A (all)

## 2) UI Intro & First Steps

- Overview of the user interface
- Profile settings
- Download simstack.toml
- Start the runner 
- Health Tab
- Submit Tab
- Run the adder workflow 
- View the results 
- Amend basic_operations.py with a add_multiply workflow for (a+b)*c
- Run create_node_table again 
- Restart the runner
- Submit the add_multiply workflow
- Inspect the results, including the child nodes 

## 3) Machine Learning 

### 3.1 Upload data, initial analysis

- Motivation (Powerpoint, ca. 5 Minutes)
- Read the stress-strain-dataset (Datasets -> Numpy -> steel_stress_strain_curves2.npy)
- Run the plot_one_curve node from public/plot_one_curve.py
- Inspect the results, let people play with chart options

### 3.2 Create the training data 

- explain public/extract_stress_strain_features.py
- based on public/plot_one_curve.py, motivate people to write private/analzye_one_curve.py
- explain why it is useful to have: models/stress_strain_model.py
- have poeple run analyse_one_curve.py
- use the chart inspection to check that the right data is extracted
- explain the training data approach, go over the code in public/make_training_data.py
- run public/make_training_data.py, monitor the progress while running (this is hundres od jobs)

### 3.3 Train the ML model
- motivate and explain the ML workflow public/ml_training_helper.py
- run public/train_impurity_model, note the bad correlation for S and P
- experiment: train only one species ? (models: ElementSelector)
- analyze the training data, public/visualize_training_data.py
- realize the S & P concentrations are smaller: use scaling 
- make new nodes based on private/train_impurity_model.py
- run and test the new nodes 

### 3.3. Summary and Outlook (WW)
- Dynamics Artifacts 
- Semantic data types 
- Data Privacy / PMD Simstack Instance
- Q&A

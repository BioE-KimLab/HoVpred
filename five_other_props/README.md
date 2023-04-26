# Predictive models for five other properties

- data/ : Flash points, liquid Cp, and melting points from DIPPR were redacted, other values from the supporting information of literature are available

- best_211007/ : The trained weights of the HoV model to carry out transfer learning

- results_/ : The resulting model files 
-- results_/*_models : 20 models * (0 - 6 transferred layers during the transfer learning) = 140 models per each property
-- results_/*_best : The best model for each property which was used for screening new fuel and working fuel candidates

- Example commands (training and prediction, for prediction - 'molecules_to_predict.csv' file is needed as an input):
```python

python main.py -num_layers_transfer 0 -modelname tc_notfl_0 -random_state 0 -prop Tc
python main.py -num_layers_transfer 2 -fix_transferred_weights -num_layers_to_fix 2 -modelname tc_tfl_2_0 -random_state 0 -prop Tc
python main.py -predict -modelname FP_best

```

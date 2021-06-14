# Dependencies
numpy
matplotlib
torch

# Training a Transformer
Two scripts are given for training transformers:
You can use generic_run.sh to train a single transformer, e.g.:
```
lang=reverse_100 heads=1 layers=2 dim=50 epochs=20 dropout=0.1 gamma=0.99 lr=0.0003 subfolder=example ./generic_run.sh
````

Or alternately, run an experiment akin to that of Table 1, using, e.g.:
```
lang=reverse_100 heads=1 layers=2 ./multirun.sh
```
This script has multiple further parameters inside, e.g., number of epochs, initial learning rate, and subfolder to store the transformers in (initially, `example`).

Both of these scripts will create one or more (depending on script) transformer(s) in the folder `lms/[subfolder]/[language name]/[generated identifier for your new transformer]`. You can keep an eye on training in a `training_prints.txt` file that will be created there, this is also where the final train, validation, and test accuracy will be printed at the end.
You can cut training short without losing everything by sending a keyboard interrupt (Ctrl+C on a Mac or Linux). It will print that training has been interrupted, save the model as it was at its best validation accuracy, and move to evaluation of the trained model. 

These scripts train transformers, storing the resulting models and their accuracies in a new subfolder, `lms`. The scripts have further options, you can open them to see.

These scripts do not print the heatmaps of these transformers.

#  Languages/Tasks
You can train a transformer on any language defined and saved to the minilangs container in `Minilangs.py` (see the file for examples). You may define languages with and without attention supervision: the `reverse` language does not have attention supervision, but `sort_by_freq` does. 

You can see all of the languages by opening python3 and running:
```
import Minilangs
Minilangs.minilangs.keys()
```
additionally, you can get a feel for a specific language by peeping it:
```
Minilangs.minilangs.peep_lang("reverse_26",subset="train",start=0,stop=10)
```



# Printing the Heatmaps
To print the heatmap (and test accuracy) of a transformer, use `LoadAndDrawAttention.py`. For example:
```
python3 LoadAndDrawAttention.py --lang=reverse_100 --sequence=abcde --from-subfolder=example
```
This script also has some additional options, you can open it for details.
The heatmap will be generated in `maps/[lang]/[sequence]/[subfolder]/[the identifier of the loaded transformer]/distributions`, and the script will additionally report what the output ended up being, as well as the full path of the selected model and what test set accuracy it reached.
If the transformer has not finished training yet (or its training was cut short), but some checkpoint exists, then the script will run on this checkpoint, and report a test accuracy of 0.



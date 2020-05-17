# NLPer

Software designed to summarize texts in polish language.

It's a command-line app which also can be used as a library.

Project documentation available at [nlper.readthedocs.io](https://nlper.readthedocs.io/)

## Features

### Clean single text
Tool for cleaning and lemmatization on provided text

### Clean data frames
Tool for cleaning and lemmatization on all the texts in dataframe.
Additionally filters texts below minimum threshold and trims texts above the maximum threshold.

### Train model
Allows to train or fine-tune deep learning model

### Summarize text
Tool which summarizes the provided text

### Split dataframes into train / test / validation parts
Tool for splitting cleaned dataframes into train / test and validation parts before training

## Installation

With docker

```bash
docker build -t NLPer . && docker run -it NLPer
```

Please keep in mind the docker building might take up to 45 minutes.  

### Without docker

Create virtualenv

``` python
$ python3.7 -m venv .nlper-venv
$ source .nlper-venv/bin/activate
```

Install application

``` python
(.nlper-venv) $ pip install .
```

In order to work with SpaCy it is necessary to download [pl_spacy_model 0.1.0](http://zil.ipipan.waw.pl/SpacyPL?action=AttachFile&do=get&target=pl_spacy_model-0.1.0.tar.gz) 
Then please extract the file and install using

```bash
cd pl_spacy_model-0.1.0
pip install .
``` 



## Usage

### Clean single text

Command-line interface:

Please add text as an argument:

``` python
(.nlper-venv) $ clean-text "Wikipedia – wielojęzyczna encyklopedia internetowa działająca zgodnie z zasadą otwartej treści."
wikipedia wielojęzyczny encyklopedia internetowy działać zgodnie z zasada otwarty treść .
```

### Clean dataframes

Command-line interface:

Please add the path to appropriate config for `clean-data`

``` python
(.nlper-venv) $ train config/dataframe_cleaner.yaml

```

Example config for `clean-data`:

```yaml
input: 'input_dataframe_path'
output: 'output_dataframe_path'

# reducing dataframe
columns_to_skip: ['url']
columns_to_merge_as_text: ['text', 'text_list', 'text_main_points']
columns_to_merge_as_summary: ['title', 'lead']

# reduced dataframe saver
save_reduced: False
reduced_output_name: 'reduced_data'
reduced_merge_data: False
reduced_output_type: 'pickle'

# cleaning dataframe
hide_numbers: True
number_replacement: '<num>'
lemmatize: True

# cleaned dataframe saver
save_cleaned: True
cleaned_output_name: 'cleaned_all_data'
cleaned_merge_data: True
cleaned_output_type: 'csv'

# trimming dataframe
trim_data: True
text_lower_length_limit: 40
text_upper_length_limit: 400
summary_lower_length_limit: 10
summary_upper_length_limit: 100

# trimmed dataframe saver
save_trimmed: True
trimmed_output_name: 'trimmed_all_data'
trimmed_merge_data: True
trimmed_output_type: 'csv'
```

### Train model
Command-line interface:

Please add the path to appropriate config for `train`

``` python
(.nlper-venv) $ train config/train_model_config.yaml

```

Example config for `train`:

```yaml
train_test_val_dir: 'path_to_directory_with_train_test_val_data'
model_output_path: 'model_output_path'
vocab_output_path: 'vocabulary_for_model_output_path'
model_name: 'name_of_the_model'

#data settings
min_frequency_of_words_in_vocab: 10
dataframes_field_names: ['text', 'summary']

#model settings
batch_size: 16
hidden_size: 256
embed_size: 128

#training setiings
epochs: 10
learning_rate: 0.01
grad_clip: 10.0
scheduler_step_size: 5000
scheduler_gamma: 0.75
save_model_after_epoch: True
save_model_every: 2000
```

### Summarize text

Command-line interface:


``` python
(.nlper-venv) $ predict "Wikipedia – wielojęzyczna encyklopedia internetowa działająca zgodnie z zasadą otwartej treści."

```


### Split dataframes into train / test / validation parts
Tool for splitting cleaned dataframes into train / test and validation parts before training


Command-line interface:

``` python
(.nlper-venv) $ split-train-test config/split_train_test.yaml
Hello, world!
```

Example config for `split-train-test`:

```yaml
input_file: 'input_dataframe_file'
output_dir: 'output_dataframe_directory'

valid: True
```

## As a library

With the app installed, you can do `import nlper` in your notebooks and use it. 
When used as a library, there's no logging output unless you configure a logger for `nlper`.


## Development

Installation in dev mode with access to tests:

``` python
(.nlper-venv) $ pip install -e .[test]
```

Run tests:

``` python
(.nlper-venv) $ pytest

tests/integration/test_app.py::test__app__works_without_errors PASSED 
... 

====================== 1 passed in 0.02 seconds ======================
```
<!--
Run tests in docker:

```
$ make test

# a lot of output

====================== 1 passed in 0.02 seconds ======================
```

-->
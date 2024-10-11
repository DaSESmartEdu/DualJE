# Job Title Prediction as A Dual Task of Expertise Prediction in Open Source Software

## Environments

This code was tested using Python 3.10 and torch 2.0.1+cu118. 

## Data

* Place the preprocessed data into the ''data'' directory
  * The common user is extracted and selected  at  [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/T6ZRJT).
  * Using [GraphQL](https://docs.github.com/en/graphql), you can get users' GitHub repository information. You can extract APIs from these repository codes and leverage a dictionary of pre-trained [API expertise model](https://zenodo.org/records/4457108) for filtering and processing.
  * The user’s professional experience can be obtained [here](https://prospeo.io/api/linkedin-email-finder), the user's main job path is extracted and  [Job Market Model](https://huggingface.co/jjzha/esco-xlm-roberta-large)  is used for job standardization.

## Train dual model

* Train JM and EM, respectively

```
  cd EM
  python run_EM.py
  cd JM
  python run_JM.py
```

*  Train primal task (expertise -> title)

```
  cd ..
  python run_primal_transformer.py
```

*  Train dual task (title -> expertise embedding)

```
  python run_dualmodel_transformer.py
```

* Train dual learning

```
  python run_dual_main.py
```

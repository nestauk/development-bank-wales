# Measuring and Predicting Property Upgradability

Last updated: 20 July 2022 by Julia Suter

## Table of Content<a id='top'></a>

[**Introduction to Project**](#intro)

- [Measuring and predicting property upgradability](#upgr)
- [Next steps](#next)

[**About this Codebase**](#imp)

- [Jupyter Notebooks](#notebooks)
- [Data](#data)

[**Technical Details**](#tech)

- [Setup](#setup)
- [Contributor Guidelines](#guidelines)

<br>

## Introduction to Project<a id='intro'></a>

[[back to top]](#top)

### Measuring and predicting property upgradability<a id='upgr'></a>

As one of our projects in A Sustainable Future at Nesta, we design a new retrofit loan together with the Development Bank Wales that will allow property owners in Wales to ugprade their home and make it more energy efficient (fewer carbon emissions). This gives home owners the opportunity to future-proof their homes, even if they could not pay for the upfront costs of retrofit measures all at once.

Using data science and the Energy Performance Certificate (EPC) data, we can help identify areas that would be suitable for launching a pilot for such a loan. The ideal area would have many properties that can be retrofitted/upgraded and lie within a less-deprived area.

This codebase demonstrates a method for predicting upgradability of a property (and areas) based on EPC records. We incorporate the fact that some properties have several EPC records over time, which allows us to study which upgrades have been implemented in the past.

While interested in the overall upgradability of a property, we also want to study the individual upgrade categories: roof, walls, floor, windows, lighting, main heat and hot water.

We consider a property upgradable in a specific category, if a) we can observe an upgrade in that category between the earliest and most recent record, and/or if b) an upgrade in that category is recommended by the EPC recommendations. Since we do not have multiple records or recommendations for all properties, there are limitations to the data. Nonetheless, we have attempted to build a tool for measuring and predicting upgradability.

The purpose of this tool is to highlight areas with highest potential for upgrades and potentially
identify most impactful combinations of retrofit measures. It will never be used to decide which property should get which upgrade (without further expert assessment) or who should/could apply for a retrofit loan.

Our upgradability model is based on observed upgrades and recommendations and not the actual potential of a property. As an alternative approach, we explore the unsupervised approach of Gaussian smoothing to measure the gap between competence (potential) and performance (observed).

More information can be found in this [slide deck](https://docs.google.com/presentation/d/1lcK895k0re_TpbtdKXe3jmRwj-dXGvQBkNsTaKxL2cU/edit?usp=sharing).

<br>
<br>
<div>
<img src="https://user-images.githubusercontent.com/42718928/179939383-0cab1163-dcb9-439d-a673-e98707d29188.png" width="1500"/>
</div>

<br>

### Next steps<a id='next'></a>

One of the next steps is extending the upgradability prediction model with additional upgrade categories, for example main heat, hot water, air tightness and energy. This will provide a more complete overview of the upgradability situation across properties and areas Wales.

The Gaussian smoothing could also be further explored and evaluated in order to study the difference between competence and performance.

Once a set of retrofit measures has been designed for the retrofit loan, we can fine-tune our model to the specific upgrades and identify in which area the retrofit loan would have the highest impact.

<br>

## Implementation Notes<a id='imp'></a>

[[back to top]](#top)

### Jupyter Notebooks<a id='notebooks'></a>

This codebase currently tackles three tasks, each with an assosciated Jupyter Notebook that guides you through the process. The notebooks are highly independent from each other, but were implemented in the following order.

- The notebook [Initial Upgradability Analysis](https://github.com/nestauk/development-bank-wales/blob/1_exploratory_analysis/development_bank_wales/analysis/Initial%20Upgradability%20Analysis.py) explores the potential of the EPC records and its recommendations in regards to studying upgrades and upgradability.

- The notebook [Predicting Upgradability](https://github.com/nestauk/development-bank-wales/blob/1_exploratory_analysis/development_bank_wales/analysis/Predicting%20Upgradability.py) documents how to measure and predict upgradability for a property and how to create maps showing the average upgradability per area.

- The notebook [Modelling Competence using Gaussian Smoothing](https://github.com/nestauk/development-bank-wales/blob/1_exploratory_analysis/development_bank_wales/analysis/Modelling%20Competence%20using%20Gaussian%20Smoothing.py) demonstrates an alternative unsupervised approach that explores that gap between competence and performance.

<br>

### Data<a id='data'></a>

The primary data source for this project are the EPC (Energy Performance Certificate) records for Wales, including EPC recommendations. Furthermore, we require geographical information and information about deprivation (WIMD).

For employees of Nesta, the data is available via the S3 bucket `asf-core-data`. The data will be retrieved automatically when executing the scripts so no manual downloading is required.

Everyone else please raise an issue if the data is required.

You can also download the input data from the S3 bucket called [asf-development-bank-wales](https://s3.console.aws.amazon.com/s3/buckets/asf-development-bank-wales?region=eu-west-2&tab=objects).

```
aws s3 sync s3://asf-development-bank-wales/wales_epc_with_recs.csv ./outputs/data/
```

<br>

## Technical Details<a id='tech'></a>

[[back to top]](#top)

### Setup<a id='setup'></a>

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`

<br>

### Contributor guidelines<a id='guidelines'></a>

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
xw

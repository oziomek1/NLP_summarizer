# Python CLI app skeleton

It's a command--line app which also can be used as a library.


## Installation

Create virtualenv

``` python
$ python3.7 -m venv .nlper-venv
$ source .nlper-venv/bin/activate
```

Install application

``` python
(.nlper-venv) $ pip install .
```

## Running application 
<!--
Check version

``` python
(.nlper-venv) $ nlper --version
0.1dev0
```
-->

Command-line interface:

``` python
(.venv) $ nlper
Hello, world!
```
<!--
Run with different logging verbosity

``` python
(.venv) $ myapp world -vv
2017-08-29 13:13:45 WARNING It's a warning
2017-08-29 13:13:45 INFO Saying hello to world
Hello, world!
```
-->

## As a library

With the app installed, you can do `import nlper` in your notebooks and use it. 
When used as a library, there's no logging output unless you configure a logger for `nlper`.


## Development

Installation in dev mode:

``` python
(.nlper-venv) $ pip install -e .[dev]
```

Run tests:

``` python
(.nlper-venv) $ pytest

tests/integration/test_app.py::test__app__works_without_errors PASSED  

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
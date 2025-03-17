# diamond💎time

DiamondTime is a season scheduler for baseball leagues. It uses [constraint programming](https://en.wikipedia.org/wiki/Constraint_programming) to generate optimal schedules based on team and field availability, preferences, and other constraints.

## Features

- Define teams, fields, weeks, and game slots
- Specify hard constraints for team and field unavailability
- Specify soft constraints for team and field preferences
- Generate balanced schedules with equal home and away games
- Output schedules in readable, JSON, or CSV formats

## Quick start

The quickest way to generate a schedule is to [get started with this Colab notebook](https://colab.research.google.com/drive/1dD5vz9uNEqlMVSiY-Tm_x-xUb3ocVzKJ).

## Installation

1. Clone the repository:

```sh
git clone https://github.com/jonathanmcmahon/diamondtime.git
cd diamondtime
```

2. Install the required dependencies:

```sh
pip install -r .
```

## Usage

### Google Colab

Launch the [notebook](https://colab.research.google.com/drive/1dD5vz9uNEqlMVSiY-Tm_x-xUb3ocVzKJ) on Colab.

### Command Line Interface

To generate a schedule: 

1. Edit the `config.yaml` and `constraints.yaml` files 

2. Call the `diamondtime.py` script:

```sh
python3 diamondtime.py config.yaml --constraints constraints.yaml
```

## License 

See [LICENSE](LICENSE).
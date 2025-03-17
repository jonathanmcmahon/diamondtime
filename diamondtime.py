import itertools
import json
import sys
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path

import pandas as pd
import rich
import typer
import yaml
from ortools.sat.python import cp_model
from pydantic import BaseModel

WINDOWSIZE = 78

app = typer.Typer(help="DiamondTime season scheduler")


class DiamondTimeError(Exception):
    """Custom exception for DiamondTime errors."""


class DiamondTimeUnsolveableError(DiamondTimeError):
    """Custom exception for unsolvable DiamondTime problems."""


class DiamondTimeConstraintError(DiamondTimeError):
    """Custom exception for DiamondTime constraint errors."""


class OutputType(StrEnum):
    readable: str = auto()
    json: str = auto()
    csv: str = auto()


class Season(BaseModel):
    teams: list[str]
    fields: list[str]
    weeks: list[str]
    slots: list[str]

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def n_weeks(self) -> int:
        return len(self.weeks)

    @property
    def n_slots(self) -> int:
        return len(self.slots)


class TeamUnavailable(BaseModel):
    team: str
    week: str
    slot: str | None = None


class FieldUnavailable(BaseModel):
    field: str
    week: str
    slot: str | None = None


class TeamPrefersNot(BaseModel):
    team: str
    week: str
    slot: str | None = None
    weight: int = 5


class FieldPreference(BaseModel):
    team: str
    field: str
    weight: int = 10


class ConstraintSpecification(BaseModel):
    team_unavailable: list[TeamUnavailable]
    field_unavailable: list[FieldUnavailable]
    field_preference: list[FieldPreference]
    team_prefers_not: list[TeamPrefersNot]


@dataclass
class HardConstraintSet:
    # List of (team, day, slot) tuples
    team_unavailable: list[tuple[int, int, int]] = field(default_factory=list)
    # List of (field, day, slot) tuples
    field_unavailable: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class SoftConstraintSet:
    # List of (team, day, slot, weight) tuples
    team_prefers_not: list[tuple[int, int, int, int]] = field(
        default_factory=list
    )
    # Map of team name to {field: weight} preference dict
    field_preferences: dict[int, dict[int, int]] = field(default_factory=dict)


class DiamondTimeScheduler:
    def __init__(
        self,
        season: Season,
        series_length: int = 2,
        output_format: OutputType = OutputType.readable,
    ):
        # Store name mappings
        self.season = season

        self.series_len = series_length
        self.output_format = output_format

        # Short name
        sn = self.season

        # Create reverse mappings for lookups
        self.team_name_to_idx = {name: i for i, name in enumerate(sn.teams)}
        self.field_name_to_idx = {name: i for i, name in enumerate(sn.fields)}
        self.week_name_to_idx = {name: i for i, name in enumerate(sn.weeks)}
        self.slot_name_to_idx = {name: i for i, name in enumerate(sn.slots)}

        # Calculate total available slots and required games
        self.total_slots = sn.n_weeks * sn.n_fields * sn.n_slots
        self.total_games_required = (
            sn.n_teams * (sn.n_teams - 1) * series_length // 2
        )

        # Check if the total number of games required exceeds the available slots
        if self.total_games_required > self.total_slots:
            raise DiamondTimeUnsolveableError(
                f"not enough slots ({self.total_slots}) to schedule all required games ({self.total_games_required})"
            )

        # Initialize constraints
        self.hard_constraints = HardConstraintSet()
        self.soft_constraints = SoftConstraintSet()

        # By default, give preference to the largest field (field 0)
        for team in range(sn.n_teams):
            # Weight of 10 for field 0
            self.soft_constraints.field_preferences[team] = {0: 10}

    def _add_hard_constraint_team_unavailable(
        self, team: str, day: str, slot: str | None = None
    ):
        """Add a hard constraint that a team is unavailable on a specific day and (optionally) slot."""
        team_idx = self.team_name_to_idx[team]
        day_idx = self.week_name_to_idx[day]

        if slot is None:
            # If slot is not specified, make the team unavailable for all slots on that day
            for s in range(self.season.n_slots):
                self.hard_constraints.team_unavailable.append(
                    (team_idx, day_idx, s)
                )
        else:
            slot_idx = self.slot_name_to_idx[slot]
            self.hard_constraints.team_unavailable.append(
                (team_idx, day_idx, slot_idx)
            )

    def _add_hard_constraint_field_unavailable(
        self, field: str, day: str, slot: str | None = None
    ):
        """Add a hard constraint that a field is unavailable on a specific day and (optionally) slot."""
        field_idx = self.field_name_to_idx[field]
        day_idx = self.week_name_to_idx[day]

        if slot is None:
            # If slot is not specified, make the field unavailable for all slots on that day
            for s in range(self.season.n_slots):
                self.hard_constraints.field_unavailable.append(
                    (field_idx, day_idx, s)
                )
        else:
            slot_idx = self.slot_name_to_idx[slot]
            self.hard_constraints.field_unavailable.append(
                (field_idx, day_idx, slot_idx)
            )

    def _add_soft_constraint_team_prefers_not(
        self, team: str, day: str, slot: str | None = None, weight: int = 5
    ):
        """Add a soft constraint that a team prefers not to play on a specific day and (optionally) slot."""
        team_idx = self.team_name_to_idx[team]
        day_idx = self.week_name_to_idx[day]

        if slot is None:
            # If slot is not specified, add preference for all slots on that day
            for s in range(self.season.n_slots):
                self.soft_constraints.team_prefers_not.append(
                    (team_idx, day_idx, s, weight)
                )
        else:
            slot_idx = self.slot_name_to_idx[slot]
            self.soft_constraints.team_prefers_not.append(
                (team_idx, day_idx, slot_idx, weight)
            )

    def _set_field_preference(self, team: str, field: str, weight: int = 10):
        """Set a team preference for a specific field."""
        team_idx = self.team_name_to_idx[team]
        field_idx = self.field_name_to_idx[field]

        if team_idx not in self.soft_constraints.field_preferences:
            self.soft_constraints.field_preferences[team_idx] = {}

        self.soft_constraints.field_preferences[team_idx][field_idx] = weight

    def _extract_schedule(self, solver, games) -> list[dict]:
        """Extract the schedule from the solver."""
        schedule = []
        sn = self.season
        for t1 in range(sn.n_teams):
            for t2 in range(t1 + 1, sn.n_teams):
                for d in range(sn.n_weeks):
                    for f in range(sn.n_fields):
                        for s in range(sn.n_slots):
                            if (t1, t2, d, f, s) in games and solver.Value(
                                games[(t1, t2, d, f, s)]
                            ) == 1:
                                schedule.append(
                                    {
                                        "team1": t1,
                                        "team2": t2,
                                        "week": d,
                                        "field": f,
                                        "slot": s,
                                    }
                                )
                            elif (t2, t1, d, f, s) in games and solver.Value(
                                games[(t2, t1, d, f, s)]
                            ) == 1:
                                schedule.append(
                                    {
                                        "team1": t2,
                                        "team2": t1,
                                        "week": d,
                                        "field": f,
                                        "slot": s,
                                    }
                                )
        return schedule

    def _schedule_to_dataframe(self, schedule: list[dict]) -> pd.DataFrame:
        """Return the schedule as a dataframe."""
        # Convert to dataframe
        df = pd.DataFrame(schedule)

        # Sort schedule by week, slot, and field
        df = df.sort_values(by=["week", "slot", "field"]).reset_index(drop=True)

        # Convert indices to names
        df["team1"] = df["team1"].apply(lambda x: self.season.teams[x])
        df["team2"] = df["team2"].apply(lambda x: self.season.teams[x])
        df["week"] = df["week"].apply(lambda x: self.season.weeks[x])
        df["field"] = df["field"].apply(lambda x: self.season.fields[x])
        df["slot"] = df["slot"].apply(lambda x: self.season.slots[x])

        # Rename columns for better readability
        df.rename(
            columns={
                "team1": "home",
                "team2": "away",
                "week": "week",
                "field": "field",
                "slot": "slot",
            },
            inplace=True,
        )

        # Reorder columns
        df = df[["week", "slot", "home", "away", "field"]]

        return df

    def add_constraints(self, constraints: ConstraintSpecification):
        """Add constraints to the scheduler."""
        try:
            # Process hard constraints
            for c in constraints.team_unavailable:
                self._add_hard_constraint_team_unavailable(
                    c.team, c.week, c.slot
                )

            for c in constraints.field_unavailable:
                self._add_hard_constraint_field_unavailable(
                    c.field, c.week, c.slot
                )

            # Process soft constraints
            for c in constraints.team_prefers_not:
                self._add_soft_constraint_team_prefers_not(
                    c.team, c.week, c.slot, c.weight
                )

            for c in constraints.field_preference:
                self._set_field_preference(c.team, c.field, c.weight)

        except Exception as e:
            raise DiamondTimeConstraintError(
                f"Error processing constraints: {e}"
            ) from e

    def solve(self):
        """Solve the scheduling problem and return the solution."""

        model = cp_model.CpModel()

        # Create variables
        # games[t1, t2, w, f, s] = 1 if team t1 plays team t2 on week w, field f, slot s
        games = {}
        sn = self.season

        for t1 in range(sn.n_teams):
            for t2 in range(t1 + 1, sn.n_teams):
                for d in range(sn.n_weeks):
                    for f in range(sn.n_fields):
                        for s in range(sn.n_slots):
                            games[(t1, t2, d, f, s)] = model.NewBoolVar(
                                f"game_t{t1}_t{t2}_d{d}_f{f}_s{s}"
                            )
                            games[(t2, t1, d, f, s)] = model.NewBoolVar(
                                f"game_t{t2}_t{t1}_d{d}_f{f}_s{s}"
                            )

        # Constraint: Each game slot can have at most one game
        for d in range(sn.n_weeks):
            for f in range(sn.n_fields):
                for s in range(sn.n_slots):
                    model.Add(
                        sum(
                            games.get((t1, t2, d, f, s), 0)
                            for t1 in range(sn.n_teams)
                            for t2 in range(sn.n_teams)
                            if t1 != t2
                        )
                        <= 1
                    )

        # Constraint: Each team can play at most one game per week
        for d in range(sn.n_weeks):
            for t in range(sn.n_teams):
                model.Add(
                    sum(
                        games.get((t1, t2, d, f, s), 0)
                        for t1, t2 in itertools.combinations(
                            range(sn.n_teams), 2
                        )
                        if t1 == t or t2 == t
                        for f in range(sn.n_fields)
                        for s in range(sn.n_slots)
                    )
                    <= 1
                )

        # Constraint: Each team plays against every other team exactly series_len times
        for t1 in range(sn.n_teams):
            for t2 in range(sn.n_teams):
                if t1 != t2:
                    model.Add(
                        sum(
                            games.get((t1, t2, d, f, s), 0)
                            for d in range(sn.n_weeks)
                            for f in range(sn.n_fields)
                            for s in range(sn.n_slots)
                        )
                        == self.series_len // 2
                    )

        # Constraint: Balanced home/away games - each team plays equal number of home and away games against each opponent
        for t1 in range(sn.n_teams):
            for t2 in range(t1 + 1, sn.n_teams):
                # Count games where team1 is home against team2
                team1_home_games = sum(
                    games.get((t1, t2, d, f, s), 0)
                    for d in range(sn.n_weeks)
                    for f in range(sn.n_fields)
                    for s in range(sn.n_slots)
                )

                # Count games where team1 is away against team2
                team1_away_games = sum(
                    games.get((t2, t1, d, f, s), 0)
                    for d in range(sn.n_weeks)
                    for f in range(sn.n_fields)
                    for s in range(sn.n_slots)
                )

                # Ensure equal number of home and away games
                model.Add(team1_home_games == team1_away_games)

        # Add hard constraints: Team unavailable
        for team, day, slot in self.hard_constraints.team_unavailable:
            for other_team in range(sn.n_teams):
                if team != other_team:
                    t1, t2 = min(team, other_team), max(team, other_team)
                    for field in range(sn.n_fields):
                        if (t1, t2, day, field, slot) in games:
                            model.Add(games[(t1, t2, day, field, slot)] == 0)
                        if (t2, t1, day, field, slot) in games:
                            model.Add(games[(t2, t1, day, field, slot)] == 0)

        # Add hard constraints: Field unavailable
        for field, day, slot in self.hard_constraints.field_unavailable:
            for t1 in range(sn.n_teams):
                for t2 in range(t1 + 1, sn.n_teams):
                    if (t1, t2, day, field, slot) in games:
                        model.Add(games[(t1, t2, day, field, slot)] == 0)

        # Create objective function for soft constraints
        objective_terms = []

        # Soft constraint: Team prefers not to play
        for team, day, slot, weight in self.soft_constraints.team_prefers_not:
            for other_team in range(sn.n_teams):
                if team != other_team:
                    t1, t2 = min(team, other_team), max(team, other_team)
                    for field in range(sn.n_fields):
                        if (t1, t2, day, field, slot) in games:
                            # Penalize scheduling this game
                            objective_terms.append(
                                weight * games[(t1, t2, day, field, slot)]
                            )

        # Soft constraint: Field preferences
        for t1 in range(sn.n_teams):
            for t2 in range(t1 + 1, sn.n_teams):
                for d in range(sn.n_weeks):
                    for f in range(sn.n_fields):
                        for s in range(sn.n_slots):
                            if (t1, t2, d, f, s) in games:
                                # Get field preference weights for both teams
                                t1_weight = (
                                    self.soft_constraints.field_preferences.get(
                                        t1, {}
                                    ).get(f, 0)
                                )
                                t2_weight = (
                                    self.soft_constraints.field_preferences.get(
                                        t2, {}
                                    ).get(f, 0)
                                )

                                # We want to maximize field preference, so we add a negative term
                                objective_terms.append(
                                    -1
                                    * (t1_weight + t2_weight)
                                    * games[(t1, t2, d, f, s)]
                                )

        # Set objective to minimize the sum of all penalty terms
        if objective_terms:
            model.Minimize(sum(objective_terms))

        # Solve the model
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # Check if a solution was found
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract the schedule
            schedule = self._extract_schedule(solver, games)

            # Convert to DataFrame
            schedule = self._schedule_to_dataframe(schedule)

            return schedule

        else:
            raise DiamondTimeUnsolveableError(
                "No feasible solution found for the given problem."
            )


def load_config(yaml_file: Path) -> Season:
    """Load configuration from a YAML file"""
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    s = Season(**config)

    print("🌭 Season config loaded:\n")
    rich.print(s)
    print()

    return s


def load_constraints(yaml_file: Path) -> ConstraintSpecification:
    """Load constraints from a YAML file"""

    with open(yaml_file) as f:
        c = yaml.safe_load(f)

    c = ConstraintSpecification(**c)

    print("🌭 Constraints loaded:\n")
    rich.print(c)
    print()

    return c


@app.command()
def main(
    config: Path,
    series_length: int = 2,
    constraints: Path | None = None,
    output: OutputType = "readable",
):
    print("⚾⚾⚾  DiamondTime Season Scheduler  ⚾⚾⚾")
    print()

    if series_length % 2 != 0:
        print(
            "ERROR: series_length must be even (# of home & away games must be equal)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load configuration from YAML
    try:
        season = load_config(config)
    except Exception as e:
        print(f"ERROR: could not load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Create scheduler with either explicit counts or inferred from config
    try:
        scheduler = DiamondTimeScheduler(
            season=season,
            series_length=series_length,
            output_format=output,
        )
    except DiamondTimeUnsolveableError as e:
        print(f"ERROR: could not create scheduler: {e}", file=sys.stderr)
        sys.exit(1)

    # Load constraints if provided
    if constraints:
        # Load constraints from YAML
        try:
            constraints = load_constraints(constraints)
        except Exception as e:
            print(
                f"ERROR: could not load constraints from file: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            scheduler.add_constraints(constraints)
        except DiamondTimeError as e:
            print(f"ERROR: could not process constraints: {e}", file=sys.stderr)
            sys.exit(1)

    # Solve and print the schedule
    try:
        result = scheduler.solve()
    except DiamondTimeUnsolveableError as e:
        print(f"Could not solve the problem: {e}", file=sys.stderr)
        sys.exit(1)

    if output == "json":
        print(json.dumps(result.to_dict(orient="records"), indent=2))
    elif output == "csv":
        print(result.to_csv(index=False))
    else:
        print("=" * WINDOWSIZE)
        print(f"\n{'🏟️ Season Schedule': ^{WINDOWSIZE}}\n")
        rich.print(result)
        print("\n\n")


if __name__ == "__main__":
    app()

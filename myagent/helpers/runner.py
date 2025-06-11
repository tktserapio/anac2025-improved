import time

from negmas.helpers import humanize_time
from rich import print
from scml.utils import (
    anac2024_oneshot,
    anac2024_std,
    DefaultAgentsStd2024,
    DefaultAgentsOneShot2024,
)
from tabulate import tabulate
#from scml_agents import get_agents
from other_agents.MatchingPennies import MatchingPenniesAgent as MatchingPenniesAgent
from other_agents.QuantityOriented import QuantityOrientedAgent
#from myagent.myagent import MATAgent

init_competitors= [
    MatchingPenniesAgent,
    QuantityOrientedAgent,
]



def run(
    competitors: list = None,
    competition="oneshot",
    reveal_types=True,
    n_steps=20,
    n_configs=2,
    debug=True,
    serial=False,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competitors: A list of competitor classes
        competition: The competition type to run (possibilities are oneshot, std).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles, production graphs etc
        reveal_types: If given, agent names will reveal their type (kind of) and position
        debug: If given, a debug run is used.
        serial: If given, a serial run will be used.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value
        - To use breakpoints in your code under pdb, pass both debug=True and serial=True

    """



    if competitors is None:
        competitors = []

    # Add your default init_competitors
    from other_agents.MatchingPennies import MatchingPenniesAgent
    from other_agents.QuantityOriented import QuantityOrientedAgent

    init_competitors = [
        MatchingPenniesAgent,
        QuantityOrientedAgent,
    ]

    # Merge the user-provided competitors and built-ins
    all_competitors = competitors + init_competitors




    if competition == "oneshot":
        all_competitors += list(DefaultAgentsOneShot2024)
        runner = anac2024_oneshot
    else:
        all_competitors += list(DefaultAgentsStd2024)
        runner = anac2024_std

    start = time.perf_counter()
    results = runner(
        competitors=all_competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        debug=debug,
        parallelism="serial" if serial else "parallel",
        agent_name_reveals_position=reveal_types,
        agent_name_reveals_type=reveal_types,
    )



    # just make names shorter
    print("Columns in total_scores:", results.total_scores.columns.tolist())
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
        "."
    ).str[-1]
    # display results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    print('num competitors: ',len(competitors))


if __name__ == "__main__":
    import typer

    typer.run(run)

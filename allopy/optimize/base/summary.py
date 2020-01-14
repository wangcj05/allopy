from collections import abc
from itertools import zip_longest
from typing import Tuple

import numpy as np


class Summary:
    def __init__(self, algorithm: str, prog_setup: list, opt_setup: list, bounds: Tuple[np.ndarray, np.ndarray]):
        self.alg = algorithm
        self.prog_setup = prog_setup
        self.opt_setup = opt_setup
        self.violations = []
        self.tight_hin = []
        self.solution = []
        self.lb, self.ub = bounds

    def _repr_html_(self):
        """Used when printing to IPython notebooks"""

        def wrap(tag, element):
            return f"<{tag}>{element}</{tag}>"

        setup = ""
        for l1, l2 in zip_longest(self.prog_setup, self.opt_setup):
            x1, x2 = ('', '') if l1 is None else l1
            y1, y2 = ('', '') if l2 is None else l2

            setup += wrap('tr', ''.join(wrap('td', i) for i in (x1, x2, y1, y2)))

        bounds = " ".join(
            wrap(
                'tr',
                ''.join([wrap('td', round(l, 6)), wrap('td', round(6))]))
            for l, u in zip(self.lb, self.ub))

        if len(self.violations) > 0:
            violations = ''.join(wrap('li', f"{i + 1:3d}: {n}") for i, n in enumerate(self.violations))
            results = f"""
<div>
    <b>No solution found</b>. List of constraints violated below:
    <ul>
    {violations}
    </ul>
</div>
            """
        else:
            if len(self.tight_hin) == 0:
                tight = 'None of the constraints were tight'
            else:
                tight = ''.join(wrap("li", f'{i + 1:3d}: {n}') for i, n in enumerate(self.tight_hin))
                tight = f'The following inequality constraints were tight: <br/><ul>{tight}</ul>'

            sol = self.solution
            results = f"""
<div>
    <b>Program found a solution</b>
    <p>
        Solution: [{', '.join(str(round(x, 6)) for x in sol) if isinstance(sol, abc.Iterable) else sol}]
    </p>
    {tight}
</div> 
"""

        return f"""
<h1>Base Optimizer</h1>
<h3>Algorithm: {self.alg}</h3>
<hr/>
<table>
    <tr>
        <th>Problem Setup</th>
        <th>Value</th>
        <th>Optimizer Setup</th>
        <th>Value</th>
    </tr>
    {setup}
</table>
<hr/>
<table>
    <tr>
        <th>Lower Bound</th>
        <th>Upper Bound</th>
    </tr>
    {bounds}
</table>
<hr/>
<h3>Results</h3>
{results}
        """

    def as_text(self):
        n = 84

        def divider(char='-'):
            return '\n'.join(['', char * n, ''])

        def new_lines(x=1):
            return '\n' * x

        # names
        rows = [
            f"{'Portfolio Optimizer':^84s}",
            divider('='),
            new_lines(),
            f'Algorithm: {self.alg}',
            divider(),
            f'{"Optimizer Setup":42s}{"Options":42s}',
        ]

        # optimization details
        for l1, l2 in zip_longest(self.prog_setup, self.opt_setup):
            x1, x2 = ('', '') if l1 is None else l1
            y1, y2 = ('', '') if l2 is None else l2
            rows.append(f"{x1:28s}{str(x2):>12s}    {y1:28s}{str(y2):>12s}")

        # bounds
        rows.extend([
            divider(),
            f'{"Lower Bounds":>15s}{"Upper Bounds":>15s}',
            *[f"{l:15.6f}{u:15.6f}" for l, u in zip(self.lb, self.ub)],
            new_lines(2)
        ])

        # results
        rows.extend([f'{"Results":84s}', divider()])

        if len(self.violations) > 0:
            rows.extend([
                'No solution found. List of constraints violated below: ',
                *[f"{i + 1:3d}: {n}" for i, n in enumerate(self.violations)]
            ])
        else:
            sln = ''.join(str(x) for x in self.solution) if isinstance(self.solution, abc.Iterable) else self.solution
            rows.extend([
                'Program found a solution',
                f"Solution: [{sln}]",
                new_lines()
            ])

            if (len(self.tight_hin)) == 0:
                rows.append('None of the constraints were tight')
            else:
                rows.extend([
                    'The following inequality constraints were tight: ',
                    *[f'{i + 1:3d}: {n}' for i, n in enumerate(self.tight_hin)]
                ])

        return '\n'.join(rows)

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return self.as_text()

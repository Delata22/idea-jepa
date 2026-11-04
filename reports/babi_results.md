| model | params | task | seed 0 | seed 1 | seed 2 | mean | seeds < 5 % |
|---|---|---|---|---|---|---|---|
| Idea-JEPA (minimal, ours) | 0.92 M | qa1 | 0.6 % | 0.0 % | 0.0 % | 0.2 % | 3/3 |
| Idea-JEPA (minimal, ours) | 0.92 M | qa2 | 0.6 % | 2.8 % | 18.8 % | 7.4 % | 2/3 |
| Idea-JEPA (minimal, ours) | 0.92 M | qa3 | 71.1 % | 75.7 % | 70.5 % | 72.4 % | 0/3 |
| B0 transformer | 0.93 M | qa1 | 46.2 % | 48.8 % | 48.3 % | 47.8 % | 0/3 |
| B0 transformer | 0.93 M | qa2 | 81.8 % | 81.0 % | 82.5 % | 81.8 % | 0/3 |
| B0 transformer | 0.93 M | qa3 | 84.7 % | 78.5 % | 78.5 % | 80.6 % | 0/3 |
| B3 window 16 + delta memory | 0.95 M | qa1 | 28.2 % | 28.0 % | 36.5 % | 30.9 % | 0/3 |
| B3 window 16 + delta memory | 0.95 M | qa2 | 76.3 % | 81.8 % | 79.7 % | 79.3 % | 0/3 |
| B3 window 16 + delta memory | 0.95 M | qa3 | 80.3 % | 84.4 % | - | 82.3 % | 0/2 |
| B0, no dropout | 0.93 M | qa1 | 49.4 % | 51.3 % | 52.8 % | 51.2 % | 0/3 |
| B0, no dropout | 0.93 M | qa2 | 83.6 % | 81.3 % | 82.2 % | 82.4 % | 0/3 |
| B3, no dropout | 0.95 M | qa1 | 19.7 % | 27.0 % | 29.6 % | 25.4 % | 0/3 |
| B3, no dropout | 0.95 M | qa2 | 81.6 % | 80.2 % | 79.5 % | 80.4 % | 0/3 |

qa3, Idea-JEPA: test error by number of times the object moved

| seed | 1 move | 2 moves | 3+ moves |
|---|---|---|---|
| 0 | - | 70.3 % (n=266) | 71.4 % (n=734) |
| 1 | - | 79.7 % (n=266) | 74.3 % (n=734) |
| 2 | - | 66.2 % (n=266) | 72.1 % (n=734) |

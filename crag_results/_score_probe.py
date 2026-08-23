import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_crag import fold_numbers, gold_in_text, pred_matches_gold, _large_number_close

g = ["222104681"]
p1 = "Wild Wild West made $222.1 million at the box office across the globe."
p2 = "Wild Wild West made $222,104,681 sales at the box office across the globe."
print("fold1", fold_numbers(p1))
print("fold2", fold_numbers(p2))
print("gold_in 1", gold_in_text(g, p1))
print("gold_in 2", gold_in_text(g, p2))
print("large1", _large_number_close(g, p1))
print("pred1", pred_matches_gold(g, p1))
print("pred2", pred_matches_gold(g, p2))
print(
    "zak",
    pred_matches_gold(
        ["3"],
        'Zak Santiago and Brenda Crichlow played together in 2 movies: "Turbulence 3: Heavy Metal" (2001).',
    ),
)
print("ross", pred_matches_gold(["3"], "Ross Geller got divorced three times."))
print("kelly", pred_matches_gold(["5"], "Kelly Oubre Jr. has played for 5 teams."))

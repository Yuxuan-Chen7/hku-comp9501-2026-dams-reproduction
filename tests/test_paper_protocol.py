"""Protocol contract transcribed from the rendered PDF, Table III (page 9).

The PDF text layer interleaves its two columns. Read the visual table, not
the extracted token order, when deciding which subcategory pair is target.
"""
from src.data import TASKS


def test_domains_match_the_visual_columns_in_paper_table_iii():
    targets={
        'rec_vs_sci':('rec.sport.hockey','sci.space'),
        'comp_vs_rec':('comp.sys.mac.hardware','rec.sport.hockey'),
        'sci_vs_comp':('sci.space','comp.sys.mac.hardware'),
    }
    for task,target in targets.items():
        assert TASKS[task][3]==target
        categories=[c for pair in TASKS[task] for c in pair]
        assert len(categories)==len(set(categories))==8

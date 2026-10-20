"""Diff de código entre prototipo original y adaptadores revisados, sin tocar fuentes."""
import difflib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'03_prototype_v31'


def main():
    pairs=[('v31_identity.py','v31_revision_identity.py'),
           ('v31_handoff.py','v31_revision_handoff.py'),
           ('01_run_shadow.py','02_run_revision.py')]
    chunks=['# Diff comparativo de módulos. La revisión es un adaptador independiente, NO un parche aplicado a V3/V3.1.\n']
    for before,after in pairs:
        left=(OLD/before).read_text(encoding='utf-8').splitlines(keepends=True)
        right=(HERE/after).read_text(encoding='utf-8').splitlines(keepends=True)
        chunks+=list(difflib.unified_diff(left,right,fromfile='original/'+before,tofile='revision/'+after,n=3))
    (HERE/'revision_code_diff.patch').write_text(''.join(chunks),encoding='utf-8')
    print('diff bytes',(HERE/'revision_code_diff.patch').stat().st_size)


if __name__=='__main__':main()

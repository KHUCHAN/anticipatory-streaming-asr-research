"""Builds a self-contained Slurm script: replaces __NAME__ markers in a template with the contents of hpc/<name>.*."""
import re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def pack(template):
    text=(HERE/'hpc'/template).read_text()
    return re.sub(r'^__([A-Z_]+)__$',lambda m:next(p for p in sorted((HERE/'hpc').glob(m.group(1).lower()+'.*')) if p.suffix!='.sh').read_text().rstrip('\n'),text,flags=re.M)
if __name__=='__main__':sys.stdout.write(pack(sys.argv[1]))

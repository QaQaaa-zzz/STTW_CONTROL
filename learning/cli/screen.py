#!/usr/bin/env python3
"""Run a declared baseline-only development screening budget."""
import argparse
from sttw_control.screening import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('task','protocol','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();run(a.task,a.protocol,a.output)

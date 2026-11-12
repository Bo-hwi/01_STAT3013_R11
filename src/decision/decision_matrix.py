"""Runnable M4 official-source Decision Matrix entry point (post-run migration)."""
from .final_handoff import main, make_tables, recommendation

__all__=['main','make_tables','recommendation']

if __name__=='__main__':main()

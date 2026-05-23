"""DEPRECATED — retained for the legacy eval/run_eval.py harness only.

Phase F removed Open WebUI from the deployed stack. The retrieval engine and
the analytics queries that used to live here have been ported into the
`rag_api` package (see `rag_api/retrieval.py` and `rag_api/analytics.py`),
which is now what the dashboard and any new code should import.

Do not add new code here. When eval/run_eval.py is rewritten against the
rag-api HTTP surface, this package can be deleted.
"""

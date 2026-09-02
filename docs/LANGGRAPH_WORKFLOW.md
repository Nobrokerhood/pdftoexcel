# LangGraph Workflow

The first agentic workflow supports only:

- `MEMBER_RECEIPT`
- `VENDOR_INVOICE`

Graph modules:

- `app/workflows/state.py`
- `app/workflows/routing.py`
- `app/workflows/accounting_graph.py`

Implemented graph:

1. PREPARE
2. EXTRACT
3. VERIFY
4. REPAIR_EXTRACTION when verification fails and retry budget remains
5. MAP
6. VALIDATE
7. HUMAN_REVIEW

Human review is mandatory. The graph stops before final XLSX generation. Approval
is a separate explicit user action that resumes the completion path.

No ERP posting is implemented.
